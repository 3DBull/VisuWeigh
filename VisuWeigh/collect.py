"""
This script collects images and data from cattle auction sources online.
Data is stored in the following folder structure:

    -> cattle_data/
        -> raw
            {auction}_{date}_{extension}.json
            -> img/
                -> {auction}/
                    im_{#}.png

 The scraper uses multiprocessing to enable faster data collection.
 If the IMAGE_CAP_DELAY is a shorter time period, the NUM_WORKERS count may need to be increased to keep up with
 predicting and saving.

 Derek Syme
 2021-11-13
 """

import logging
import queue
import datetime
from selenium import webdriver
from selenium.common.exceptions import NoSuchWindowException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller
from time import sleep
from selenium.webdriver import ActionChains
import json
import os
import sys
import cv2
from VisuWeigh.lib.yolo import Predictor
import time
import multiprocessing as mp
from VisuWeigh.lib import paths

LOGGER = logging.getLogger(__name__)

### SETTINGS ###
IMAGE_CAP_DELAY = 1  # seconds
AUCTION_TIMEOUT = 600  # seconds
LOT_TIME_LIMIT = 10  # seconds
DATABASE_LOCATION = paths.DATABASE_LOCATION
COUNT_COWS = True
COW_DETECTION_THRESH = 0.01
NUM_WORKERS = 3
EXTEND_FILE_NAME = True
#################


if COUNT_COWS:
    predictor = Predictor(paths.YOLO, obj_thresh=COW_DETECTION_THRESH)


def _initialize_collection(url, driver, data_file_name):

    # OPEN PAGE
    driver.get(url)
    sleep(2)
    view_button = driver.find_element(By.XPATH, '//input[@value="View Only"]')
    view_button.click()

    element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "XCIRAenterbutton")))

    driver.find_element(By.ID, "XCIRAenterbutton").click()

    main_page = driver.current_window_handle

    # create data file
    if not os.path.exists(os.path.join(DATABASE_LOCATION, data_file_name)):
        with open(os.path.join(DATABASE_LOCATION, data_file_name), "w") as file:
            json.dump([], file)

    # give some time for javascript page to load
    sleep(10)

    # changing the handles
    auction_page = None
    # LOGGER.debug(driver.window_handles)
    for handle in driver.window_handles:
        if handle != main_page:
            auction_page = handle

    if auction_page is None:
        raise Exception("could not get javascript page!")

    # work on the auction page now
    driver.switch_to.window(auction_page)

    # wait for the javascript page to load
    WebDriverWait(driver, 60).until(EC.text_to_be_present_in_element((By.CLASS_NAME, 'entry_label'), 'Owner'))

    # get video container
    vid = driver.find_element(By.ID, 'amd/display/MediaContainer_0')

    # click it to get rid of audio logo
    vid.click()

    # this is the element we want to take a screenshot of
    vid = driver.find_element(By.ID, 'player')

    # move the mouse so the player bar goes away
    action = ActionChains(driver)
    action.move_to_element(driver.find_element(By.CLASS_NAME, 'lot_info')).perform()

    # wait for auction to start
    data_point = ['']
    LOGGER.info("Waiting for auction to start...")
    while data_point[0] == '':
        data_point = driver.find_elements(By.CLASS_NAME, 'data-label')
        sleep(2)

    LOGGER.info('Auction is starting... collecting data')

    return vid, data_point


def _save_data(save_q, q_lock, f_lock, sale, file_name):
    global DATABASE_LOCATION, predictor

    # save the datapoint
    while True:
        # get prediction from buffer pipe
        try:
            q_lock.acquire()
            dp = save_q.get(timeout=IMAGE_CAP_DELAY * 5)
            image = save_q.get(timeout=IMAGE_CAP_DELAY * 5)
            q_lock.release()
        except ValueError:
            LOGGER.info('Done Saving!')
            break
        except queue.Empty:
            q_lock.release()
            LOGGER.info('Worker done saving!')
            break

        # predict cow info
        if COUNT_COWS:
            try:
                #with tf.device('/cpu:0'):
                pred = predictor.predict_cow_info(image=image)
            except ValueError:
                pred = ['IMAGE_ERROR']
        else:
            pred = []

        # add prediction to datapoint
        dp.append(pred)
        dp.append(sale.value.decode("utf-8"))

        # save info
        entry = {'lot': dp[0],
                 'IMG_ID': dp[1],
                 'Type': dp[2],
                 'Shrink': dp[4],
                 'Age': dp[5],
                 'Avg_Weight': dp[6],
                 'Tot_Weight': dp[7],
                 'Hauled': dp[8],
                 'Weaned': dp[9],
                 'Feed': dp[10],
                 'Health': dp[11],
                 'Timestamp': dp[12],
                 'prediction': dp[13],
                 'auction': dp[14]
                 }
        try:
            f_lock.acquire()
            with open(os.path.join(DATABASE_LOCATION, file_name.value.decode("utf-8")), "r+") as file:
                data = json.load(file)
                data.append(entry)
                LOGGER.info('Total entries in dataset: {}'.format(len(data)))
                file.seek(0)
                json.dump(data, file)
            f_lock.release()
            LOGGER.info('Saved data: {}'.format(dp))

        except PermissionError as ex:
            file.close()
            f_lock.release()
            LOGGER.exception(f'Lock error: {ex}')
        except Exception as ex:
            f_lock.release()
            LOGGER.exception(f'Lock error: {ex}')

    save_q.close()


def main(argv):
    last_run = 0
    url = argv[0]
    auction_sale = url.split('/')[-1]

    if len(argv) != 0:
        auction_sale = argv[0]

    def end():
        # exit selenium
        driver.quit()

    # Create new data file every execution
    data_file_name = auction_sale + '_' + datetime.datetime.now().strftime('%Y-%m-%d') + '.json'

    # add extension to filename if it already exists
    if EXTEND_FILE_NAME:
        ext = 0
        while os.path.exists(os.path.join(DATABASE_LOCATION, data_file_name)):
            ext += 1
            data_file_name = auction_sale + '_' + datetime.datetime.now().strftime('%Y-%m-%d') + '_' + str(ext) + '.json'

    # check chrome driver version and install if necessary
    chromedriver_autoinstaller.install()
    driver = webdriver.Chrome()

    # if the auto installer doesn't work you can install the latest driver for your
    # version of chrome from here: https://chromedriver.chromium.org/

    # wait for driver (use for slower connections)
    #driver.implicitly_wait(10)

    # lets get this auction started!
    try:
        vid, data_point = _initialize_collection(url, driver, data_file_name)

    except Exception as ex:
        LOGGER.exception("Could not get element")
        LOGGER.exception(ex)
        end()

    # set up multiprocessing
    save_q = mp.Queue()
    save_lock = mp.Lock()
    file_lock = mp.Lock()
    vol_auction_sale = mp.RawArray('c', bytes(auction_sale, encoding='utf8'))
    vol_file_name = mp.RawArray('c', bytes(data_file_name, encoding='utf8'))
    worker = []
    for i in range(NUM_WORKERS):
        worker.append(mp.Process(target=_save_data, args=(save_q, save_lock, file_lock, vol_auction_sale, vol_file_name)))
        worker[i].start()

    lot_num_im_count = 0
    old_lot_no = ''
    record = False

    # if there is no auction data for a while, the auction must be over
    while lot_num_im_count < AUCTION_TIMEOUT / IMAGE_CAP_DELAY:
        try:
            lot_num_im_count += 1

            if lot_num_im_count > LOT_TIME_LIMIT / IMAGE_CAP_DELAY and not (LOT_TIME_LIMIT == 0):
                record = False
                data_point = driver.find_elements(By.CLASS_NAME, 'data-label')

            if old_lot_no != data_point[0].text and not record:  # if there is a change in lot#
                record = True
                lot_num_im_count = 0
            else:
                old_lot_no = data_point[0].text

            # get static image ID from file
            with open(os.path.join(DATABASE_LOCATION, 'im_id_' + auction_sale + '.txt'), "r+") as file:
                im_id = int(file.read().split()[0])
                file.seek(0)
                file.write(str(im_id + 1) + ' ' + auction_sale)

            # save image
            im_path = '{}/img/{}/im_{}.png'.format(DATABASE_LOCATION, auction_sale, im_id)
            vid.screenshot(im_path)

            # package and send data to the predict process
            # labels = driver.find_elements(By.CLASS_NAME, 'entry_label')
            data_point = driver.find_elements(By.CLASS_NAME, 'data-label')
            old_lot_no = data_point[0].text
            image = cv2.imread('{}/img/{}/im_{}.png'.format(DATABASE_LOCATION, auction_sale, im_id))

            str_data_point = []
            for i in range(len(data_point)):
                str_data_point.append(data_point[i].text)

            LOGGER.debug(str_data_point)
            str_data_point[1] = str(im_id)
            str_data_point.append(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'))
            save_q.put(str_data_point)
            save_q.put(image)
            LOGGER.info('{} items in save_q'.format(save_q.qsize()))
            # break the loop if user closes the window
#            if driver.get_log('driver')[-1]['message'].startswith('chrome not reachable'):
#                LOGGER.info('User closed window')
#               break
            # time the loop
            now = time.time()
            while now - last_run < IMAGE_CAP_DELAY:
                sleep(0.01)
                now = time.time()
            LOGGER.debug('Loop execution time: {}'.format(now-last_run))
            last_run = now

        except NoSuchWindowException as ex:
            save_q.close()
            for w in worker:
                w.join()
            end()
        except Exception as ex:
            LOGGER.exception(f'Error in extraction: {ex}')

    # finished
    save_q.close()
    for w in worker:
        w.join()
    end()


if __name__ == "__main__":
    main(sys.argv[1:])
