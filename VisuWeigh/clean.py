# Clean cattle data
# This script runs through the process of cleaning the raw cattle data.
import logging
import os
import re
import sys

import tensorflow as tf
import pandas as pd
import json
from VisuWeigh.lib.yolo import Predictor
from time import time
import datetime as dt
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image
from IPython.display import display
from io import BytesIO
import datetime
import ipywidgets as widgets
from ipywidgets import *
import warnings
from tqdm import tqdm, trange
from VisuWeigh.lib import paths

LOGGER = logging.getLogger(__name__)


def _clean_numerical_data(df):
    # ## Clean Numerical Data
    # Removing commmas from thousands place
    df.Avg_Weight = df["Avg_Weight"].str.replace(",", "")
    df.Tot_Weight = df["Tot_Weight"].str.replace(",", "")

    # remove any 'lbs' suffixes from Avg_Weight and Total_Weight columns
    def remove_lbs(s):
        if s.endswith('lbs'):
            return s[:-3]
        elif s.endswith('LBS'):
            return s[:-3]
        else:
            return s

    df.Avg_Weight = df.Avg_Weight.apply(remove_lbs)
    df.Tot_Weight = df.Tot_Weight.apply(remove_lbs)

    # replace blank weights with 0's
    def rep_blank(w):
        w = str(w)
        if w == '' or w.endswith('RANGE') or w.startswith('RANGE'):
            w = '0'

        w = w.replace(',', '')
        w = w.replace(' BASE', '')
        w = w.replace('BASE ', '')
        w = w.replace('825 825', '825')

        return float(w)

    df.Avg_Weight = df.Avg_Weight.apply(rep_blank)
    df.Tot_Weight = df.Tot_Weight.apply(rep_blank)

    def fix_time(t):
        if t.find('.') == -1:
            return t + '.00'
        else:
            return t

    df.Timestamp = df.Timestamp.apply(fix_time)

    df.Timestamp = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d-%H:%M:%S.%f')

    return df

def has_image(img_id, auction):
    """
    ### Drop Data with no Images
    :param img_id: the raw image id
    :param auction: the source folder
    :return: a boolean of image existence
    """
    if os.path.exists('{}/{}/im_{}.png'.format(paths.RAW_DATA, auction, img_id)):
        return True

    return False


def adj_timing(d):
    """
    ## Correcting the Data Timing
    One of the biggest challenges in this data is the timing of information. When the auction is live, each lot has
    information that is taken from a queue and placed in the viewing screne manually by a human that is present at the auction.
    Often times the cows are allowed into the ring before the data is swiched. Sometimes the cows are let in after the
    data is switched. Occasionaly, a lot's information did not get put in the queue and thus stays incorrect for the
    duration of the viewing for the cattle lot. Sometimes these lots will have the information entered manually during
    the viewing. This means the information is significantly delayed compared to the image view.
    This function adjusts timing of weight changes in the data based on the yolo predictions of when cows are seen in the images.

    # NOTE: Only use this funciton on consecutive data. If two aucitons have been collected simultaineously,
    separate the data by auction before using this function.

    :param d: The input dataframe
    :return: dataframe
    """


    LOT_THRESH = 3  # sets the threshold for number of no-cows between lots
    time_gap = 3  # restart the lot if there are more than time_gap seconds between datapoints
    lot_start = 0
    lot_end = 0  # lot_start and lot_end are included in the lot
    prev_end = 0
    valid_weight = 0
    valid_num_cows = 0
    i = 0
    lotinc = 0

    adj_weights = d.Avg_Weight - d.Avg_Weight
    adj_num_cows = d.num_cows - d.num_cows
    lotnum = d.num_cows - d.num_cows

    pbar = tqdm(total=len(d) - LOT_THRESH)

    while i < len(d) - LOT_THRESH:

        # find a valid lot start (two no-cows in a row)
        while (d.has_cow.iloc[i:i + LOT_THRESH].any()) or not d.has_cow.iloc[i + LOT_THRESH]:
            i += 1
            if i >= len(d) - LOT_THRESH - 1:
                break
        lot_start = i + LOT_THRESH

        # find a valid lot end
        i = lot_start
        while d.has_cow.iloc[i:i + LOT_THRESH].any():

            i += 1
            if i >= len(d) - LOT_THRESH:
                break

            # end the lot if time gap is detected between points
            if (d.iloc[i + 1].Timestamp - d.iloc[i].Timestamp).seconds > time_gap:
                break

        lot_end = i - 1
        lotinc += 1

        # traverse backward through the lot to find a valid weight_change
        for j in range(lot_end, prev_end, -1):

            # the first detected change in weight is the correct weight for the lot
            if d.iloc[j].weight_change:
                valid_weight = d.iloc[j].Avg_Weight
                valid_num_cows = d.iloc[j].num_cows
                break
            else:
                valid_weight = 0

        # if it is recorded as a single-cow lot, then truncate a multi-cow start if applicable
        if valid_num_cows == 1:
            # find the transition point
            for j in range(lot_end, lot_start, -1):
                if d.iloc[j].predicted_num_cows > 2:
                    # set the transition point as the new lot_start
                    lot_start = j + 1
                    break

        # apply valid weight info to lot
        adj_weights.iloc[lot_start:lot_end + 1] = valid_weight
        adj_num_cows.iloc[lot_start:lot_end + 1] = valid_num_cows
        lotnum.iloc[lot_start:lot_end + 1] = lotinc

        # update progress bar
        pbar.update(lot_end - prev_end)

        # set next start point
        lot_start = lot_end
        prev_end = lot_end

    frame = {'img_id': d.IMG_ID, 'weight': adj_weights, 'num_cows': adj_num_cows, 'lot_num': lotnum,
             'prediction': d.adj_prediction, 'old_weight': d.Avg_Weight,
             'predicted_num_cows': d.predicted_num_cows, 'auction': d.auction, 'timestamp': d.Timestamp}

    pbar.update(1)
    pbar.close()

    return pd.DataFrame(frame)


def pick_best_pred(pred):
    """
    Only pick the best prediction if there is more than one. (occasionally there is an artifact detection of a cow)
    :param pred: YOLO prediction from data point
    :return: YOLO prediction with only the best left
    """
    if len(pred) <= 1:
        return pred

    else:
        best = pred['Prob'].astype(float).argmax()
        return pred.iloc[best]


class DataCleaner:
    
    def __init__(self, combine_files=True, dataset=paths.DATASET, input_path=paths.RAW_DATA, output_path=paths.TRAINING_DATA):
        self.COMBINE_DATA_FILES = combine_files
        self.DATSET = dataset
        self.DATABASE_LOCATION = input_path
        self.output_path = output_path
        self.output_img_path = os.path.join(output_path, 'singles')
        self.i = 0
        self.err = 0
        self.pbar = tqdm(total=0)

    def _load_raw_data(self, file_name=None):
        """
        Loads the raw data from the cleaners associated database location.
        Loads all json files from the location if COMBINE_DATA_FILES is set to true. Otherwise it will load from the
        provided filename.
        :param file_name: Only used if COMBINE_DATA_FILES is false
        :return: A pandas DataFrame
        """
        path = ''
        f_names = []
        try:

            if self.COMBINE_DATA_FILES:
                for name in os.listdir(self.DATABASE_LOCATION):
                    if name.endswith('.json'):
                        f_names.append(name)
            else:
                if file_name is None:
                    raise ValueError('Please provide a file_name to read. Alternatively, set COMBINE_DATA_FILES to True when '
                                     'initializing the cleaner')
                else:
                    f_names.append(file_name)

            frame = []
            # load the dataset
            path = os.path.join(self.DATABASE_LOCATION, f_names.pop())
            with open(path, 'r') as file:
                frame = json.load(file)

            # add any other datasets in the folder
            for name in f_names:
                path = os.path.join(self.DATABASE_LOCATION, name)
                with open(path, 'r') as file:
                    frame.extend(json.load(file))

            d = pd.DataFrame(frame)

            LOGGER.info('Found {} files.'.format(len(f_names)))
            LOGGER.info('Loaded {} datapoints!'.format(len(d)))
        except Exception as ex:
            LOGGER.exception(f'Could not load data from {path}: {ex}')
            exit(1)

        return d

    def save_df(self, d):
        """
        Saves the data to the training data location. Make sure it is good training data before calling this.
        :param d: Dataframe to save
        :return:
        """
        try:
            with open(os.path.join(self.output_path, paths.DATASET), 'w') as file:
                df_to_save = d.to_json(orient="records")
                parsed = json.loads(df_to_save)
                json.dump(parsed, file)
        except Exception as ex:
            LOGGER.exception(f'Unable to save training data to {os.path.join(self.output_path, paths.DATASET)}'
                             f'{ex}')
            return False
        return True

    def _adjust_prediction_threshold(self, d):
        """
         ## Adjusting Prediction Thresholds
         Since the prediction column includes predictions with a probability down to 1%, we can set an alternative
         threshold for the probability of the has_cow column. From observation of the predicted data, we can tell that
         the threshold needs to be different for each source location.
         Included in the following function is a filter customizing the threshold for each auction.
         The thresholds that are set for each auction location are as follows:
            * Rimbey :
            * Westlock :
            * Ponoka : 7.5%
            * Dawson Creek : 1%
            * Beaverlodge :
        """
        thresh = 0.01
        self.i += 1

        def box_filter_gt(box, coord, thresh):
            val_1, val_2 = coord
            return box[val_1][val_2] > thresh

        try:
            if len(d.prediction) < 1:
                return d.prediction
            elif d.prediction[0] == 'IMAGE_ERROR':
                self.err += 1
                return d.prediction

            u_predict = pd.DataFrame(d.prediction)

            if d.auction == 'ponoka':
                thresh = 0.075
                bbox_thresh = 173

                # include positional threshold to avoid including cows that are not in the ring
                u_predict = u_predict[u_predict.Box.apply(box_filter_gt, args=(('y', 1), bbox_thresh))]
                u_predict = u_predict[u_predict.Prob.astype(float) > thresh]
                return u_predict

            elif d.auction == 'dawson_creek':
                thresh = 0.05

            elif d.auction == 'westlock':
                thresh = 0.155

            elif d.auction == 'beaverlodge':
                thresh = 0.1

            elif d.auction == 'rimbey':
                thresh = 0.1

            # filter by Probability threshold
            u_predict = u_predict[u_predict.Prob.astype(float) > thresh]

            self.pbar.update(1)
        except Exception as ex:
            LOGGER.exception(f'Exception while adjusting the thresholds in {d}')
            LOGGER.exception(f'{d.prediction}')
            exit(1)
        return u_predict

    def save_cropped(self, d):
        """
        Save the images in the training set to a separate folder after cropping them.
        :param d: Training data
        :return:
        """
        pred = d.prediction
        XPAD = 10
        YPAD = 20

        if len(pred) >= 1:
            try:
                if type(pred) == pd.core.frame.DataFrame:
                    pred = pred.iloc[0]

                # Load the image
                if os.path.exists('{}/{}/im_{}.png'.format(paths.RAW_DATA, d.auction, d.img_id)):
                    im = cv2.imread('{}/{}/im_{}.png'.format(paths.RAW_DATA, d.auction, d.img_id))

                    # crop the image
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    im = im[max(pred['Box']['x'][1] - YPAD * 2, 0): min(pred['Box']['y'][1] + YPAD, im.shape[0]),
                         max(pred['Box']['x'][0] - XPAD * 4, 0): min(pred['Box']['y'][0] + XPAD, im.shape[1])]

                    # save the image
                    cv2.imwrite(d.path, im)

            except Exception as ex:
                LOGGER.exception(f'Exception in saving training images: {d}')
                exit(1)

        self.pbar.update(1)

    def run(self):

        df = self._load_raw_data()

        df = _clean_numerical_data(df)

        # Adjust the prediction thresholds
        # Use the cleaner's static variables to keep track of things for the apply loop
        self.i = 0
        self.err = 0
        self.pbar = tqdm(total=len(df))
        df['adj_prediction'] = df.dropna().apply(self._adjust_prediction_threshold, axis=1)
        LOGGER.warning(f'Found {str(self.err)} images with errors')

        ### Remove data by frequency
        # removes old data that had low time resolution  (this should be removed once the system has been upgraded)
        min_time = 2
        df = df[(df.Timestamp.diff().dt.seconds < min_time) | (df.Timestamp.diff().dt.seconds > 8)]

        # Only keep data that actually has a corresponding image
        df = df[df.IMG_ID.apply(has_image, args=['auction'], axis=1)]

        LOGGER.info(f'Length of dataset after removing irrelevant data: {len(df)}')

        # remove points with collision errors
        def is_error(pred):
            if len(pred) == 1:
                return pred[0] == 'IMAGE_ERROR'
            else:
                return False
        df = df[df.prediction.apply(is_error) == False]




        # ### Add Columns Describing the number of cows
        # Add columns two columns to the data set:
        # > * num_cows : describes the number of cows from the given information
        # > * predicted_num_cows : describes the number of cows in a lot predicted from the image
        # We need these columns before running the adj_prediction function
        def get_type_num(t):
            l = t.split()
            l = re.findall("\d*", t)[0]
            if len(l)==0:
                return 0
            else:
                return int(l)

        df['num_cows'] = df.Type.apply(get_type_num)
        df['predicted_num_cows'] = df.adj_prediction.apply(len)

        # ### Add has_cow column
        # If we do not wish to use a threshold...
        df['has_cow'] = df.predicted_num_cows>0


        # Diferentiate
        df['weight_change'] = ((df.Avg_Weight.diff()!=0))*5

        # Run the timing adjustment on all the auctions
        rimbey_df = adj_timing(df[df.auction == 'rimbey'])
        westlock_df = adj_timing(df[df.auction == 'westlock'])
        dawson_df = adj_timing(df[df.auction == 'dawson_creek'])
        ponoka_df = adj_timing(df[df.auction == 'ponoka'])
        beaverlodge_df = adj_timing(df[df.auction == 'beaverlodge'])



        # Extract Only Single Cow Lots <a class="anchor" id="extractsingle"></a>
        # we can concatinate the data now that the timing is shifted already.
        # start with getting single cows
        singles_df = pd.concat([rimbey_df[rimbey_df.num_cows == 1],
                               westlock_df[westlock_df.num_cows == 1],
                               dawson_df[dawson_df.num_cows == 1],
                               ponoka_df[ponoka_df.num_cows == 1],
                               beaverlodge_df[beaverlodge_df.num_cows == 1]
                               ], ignore_index=True, sort=False)


        # all points should only have one prediction so pick the best one
        singles_df.prediction = singles_df.prediction.apply(pick_best_pred)

        LOGGER.info("Singles dataframe length: {}".format(len(singles_df)))
        LOGGER.info("Original dataframe length: {}".format(len(df)))


        # ## Generate Training Data <a class="anchor" id="generatetraining"></a>
        # We will use all the single cow lots for training data.
        # For the training dataframe we need to have the path to each image included in the dataframe.
        # Let's build the dataframe with the path included.
        # For fitting the network we only need the 'weight' column and the 'path' column, but we will use the 'prediction'
        # column information in this section to get the cropped image of one cow.
        # These images are saved in the `training/singles` folder

        # Make the output directory if it does not exist
        if not os.path.exists(os.path.join(self.output_path)):
            os.mkdir(os.path.join(self.output_path))

        # create a path column
        def get_output_path(d):
            return '{}/{}_{}_{}.png'.format(self.output_img_path, d.auction, d.lot_num, d.img_id)
        singles_df['path'] = singles_df.apply(get_output_path, axis=1)

        # save the training data
        self.save_df(singles_df.drop(columns=['img_id', 'num_cows', 'old_weight', 'predicted_num_cows']))

        # save the cropped images
        self.pbar = tqdm(total=len(singles_df))
        singles_df.apply(self.save_cropped, axis=1);

        return singles_df


class MultiViewGenerator:
    """
    ## Generate Two View Training Data

    ### Extract Two Views From Single Lots
    The generator performs extraction on single cow data.
    The purpose is to pair images representing a side view and an end view of the same animal. This is important for
    giving the CNN a better "view" of the cows weight. It uses a scoring function to rate pairs of images based on how
    well they serve as an end view and side view pair. Only the pairs of images that score better than a given threshold
    will be used in the output data.
    """

    def __init__(self, score_thresh=0.75,
                 input_path=paths.RAW_DATA,
                 output_path=os.path.join(paths.TRAINING_DATA, 'two_views'),
                 save_name='two_view_data.json'):

        self.score_thresh = score_thresh
        self.input_path = input_path
        self.output_path = output_path
        self.save_name = save_name

    # extract side-view and end-view of cow

    # returns a height/width ratio for the predictoin box
    def get_ratio(self, pred):

        width = pred['Box']['y'][0] - pred['Box']['x'][0]
        height = pred['Box']['y'][1] - pred['Box']['x'][1]
        return height / width


    def best_in_lot(self, d, lot_start, lot_end):
        """
        Finds pairs of end views and side views in a given lot that are above a given scoring threshold.
        The threshold for scoring can be set when the generator is initialized.

        :param d: Input DataFrame
        :param lot_start: The index of the dataframe where the search should begin
        :param lot_end:  The index of the dataframe where the search should end

        :return: The index of the best scoring datapoint in the lot
        """
        w1, w2 = (0.25, 0.5)  # weights for accuracy and ratio respectively

        # functions to find the best tall and short image
        score_t = lambda a, r: a * w1 + r * w2
        score_s = lambda a, r: a * w1 + w2 / r  # we invert the ratio for finding the best short image

        best_pairs = [[], []]
        scores = [[], []]
        limit = 600
        n = 0

        # keep adding the next best pair until a threshold is reached
        while n < limit:

            n += 1

            best_score = [0, 0]
            best_i = [0, 0]

            # iterate through lot to find best pair
            for i in range(lot_start, lot_end):
                # normalize prediction
                pred = d.iloc[i].prediction

                # skip this image if there is no predicted cow
                if len(pred) < 1:
                    continue

                if type(pred) == pd.DataFrame:
                    pred = pred.iloc[0]

                # find best tall image
                score = score_t(float(pred['Prob']), self.get_ratio(pred))

                if (score > best_score[0]) and not (i in best_pairs[0]) and not (i in best_pairs[1]) and not (
                        d.iloc[i].predicted_num_cows > 2):
                    best_i[0] = i
                    best_score[0] = score

                # find best short image
                score = score_s(float(pred['Prob']), self.get_ratio(pred))

                if (score > best_score[1]) and not (i in best_pairs[1]) and not (i in best_pairs[0]) and not (
                        d.iloc[i].predicted_num_cows > 2):
                    best_i[1] = i
                    best_score[1] = score

            # print(best_pairs)
            # check the average score for the pair
            if (best_score[0] + best_score[1]) / 2 >= self.score_thresh:
                best_pairs[0].append(best_i[0])
                best_pairs[1].append(best_i[1])
                scores[0].append(best_score[0])
                scores[1].append(best_score[1])

            else:
                break

        return best_pairs  # , scores


    def extract_views(self, d):
        """
        Function to perform the extraction of the best two views of one cow.
        This is done for every lot in the dataset.

        :param d: Input DataFrame
        :return: Multiview dataframe with columns: 'side' and 'end'
        """

        XPAD = 10
        YPAD = 20

        pbar = tqdm(total=len(d))
        lot_start = 0
        lot_end = 0
        prev_end = 0
        i = 0
        lotnum = 0
        cleaned_db = []

        while i < len(d) - 1:
            try:
                # find lot start
                while (i < len(d)) and (d.iloc[i].weight == 0):
                    i += 1

                lot_start = i

                # find lot end
                while (i < len(d)) and (d.iloc[i].weight > 0):
                    i += 1
                    # print(i)

                lot_end = i

                # ignore single-point lots
                if lot_end - lot_start < 1:
                    continue

                lotnum += 1

                # get the best predicitons for the current lot
                btalls, bshorts = self.best_in_lot(d, lot_start, lot_end)

                for pairnum, (btall, bshort) in enumerate(zip(btalls, bshorts)):
                    tall = d.iloc[btall]
                    short = d.iloc[bshort]
                    tall_pred = tall.prediction
                    short_pred = short.prediction
                    if type(tall_pred) == pd.core.frame.DataFrame:
                        tall_pred = tall_pred.iloc[0]
                    if type(short_pred) == pd.core.frame.DataFrame:
                        short_pred = short_pred.iloc[0]

                    im_tall = cv2.imread('{}/{}/im_{}.png'.format(self.input_path, tall.auction, tall.img_id))
                    im_short = cv2.imread('{}/{}/im_{}.png'.format(self.input_path, short.auction, short.img_id))

                    # extract the two sub-images (padded)
                    im_tall = im_tall[max(tall_pred['Box']['x'][1] - YPAD * 2, 0): min(tall_pred['Box']['y'][1] + YPAD,
                                                                                       im_tall.shape[0]),
                              max(tall_pred['Box']['x'][0] - XPAD * 4, 0): min(tall_pred['Box']['y'][0] + XPAD,
                                                                               im_tall.shape[1]), :]

                    im_short = im_short[
                               max(short_pred['Box']['x'][1] - YPAD * 2, 0): min(short_pred['Box']['y'][1] + YPAD,
                                                                                 im_short.shape[0]),
                               max(short_pred['Box']['x'][0] - XPAD, 0): min(short_pred['Box']['y'][0] + XPAD,
                                                                             im_short.shape[1]), :]

                    # reduce the depth of the image to greyscale
                    im_tall = cv2.cvtColor(im_tall, cv2.COLOR_BGR2GRAY)
                    im_short = cv2.cvtColor(im_short, cv2.COLOR_BGR2GRAY)

                    if not os.path.exists(os.path.join(self.output_path)):
                        os.mkdir(os.path.join(self.output_path))

                    # package data and append to database
                    cv2.imwrite('{}/{}_{}_{}_side.png'.format(self.output_path, lotnum, pairnum, short.img_id),
                                im_short)
                    cv2.imwrite('{}/{}_{}_{}_end.png'.format(self.output_path, lotnum, pairnum, tall.img_id),
                                im_tall)

                    entry = {'date': short.timestamp.date(), 'weight': short.weight, 'side_id': short.img_id,
                             'end_id': tall.img_id,
                             'side_path': '{}/{}_{}_{}_side.png'.format(self.output_path, lotnum, pairnum,
                                                                                  short.img_id),
                             'end_path': '{}/{}_{}_{}_end.png'.format(self.output_path, lotnum, pairnum,
                                                                                tall.img_id)}
                    cleaned_db.append(entry)

                # update loop
                pbar.update(lot_end - prev_end)
                prev_end = lot_end
                i = lot_end + 1

            except Exception as ex:
                print(i)
                print(d.iloc[i])
                print(d.iloc[i].prediction)
                raise ex

        cleaned_db = pd.DataFrame(cleaned_db)

        # save the data
        with open(os.path.join(self.output_path, self.save_name), 'w') as file:
            df_to_save = cleaned_db.to_json(orient="records")
            parsed = json.loads(df_to_save)
            json.dump(parsed, file)
        pbar.update(len(d) - lot_end)
        pbar.close()

        return cleaned_db


    def run(self, df):
        return self.extract_views(df)


def usage():

    print("""
Usage: 
    Param 1 - out_type: can be either 'singles' or 'multiview'
        'Multiview' will generate single cow data as well as multiview data.
        
    Param 2 - out_path: the path where the output data is to be stored.
    
    See the System Architecture documentation for information on where data should be located.
    """)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        usage()
    else:

        out_type = sys.argv[1]
        out_path = sys.argv[2]

        if out_type == 'singles':
            cleaner = DataCleaner(output_path=out_path)
            singles = cleaner.run()
            exit(0)

        elif out_type == 'multiview':
            cleaner = DataCleaner(output_path=out_path)
            singles = cleaner.run()
            multi_view_gen = MultiViewGenerator()
            multi_view_gen.run(singles)
            exit(0)

        else:
            usage()
