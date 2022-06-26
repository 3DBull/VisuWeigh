import logging
import os
import pandas as pd
from VisuWeigh.lib.util import evaluate_model, load_all_json
from VisuWeigh.lib import paths

LOGGER = logging.getLogger(__name__)


def load_eval_data(folder_name):
    # load evaluation data (singles)
    df = load_all_json(os.path.join(paths.EVALUATION_DATA, folder_name))
    return df


def main():
    try:
        # Retrieve the names for every set (directory) of evaluation data
        eval_names = os.listdir(paths.EVALUATION_DATA)

        # Retrieve the data
        eval_sets = [load_eval_data(name) for name in eval_names]

        # Retrieve the names of the models to be evaluated
        model_names = os.listdir(paths.TRAINED_MODELS)

        # Create a csv for storing the results if there isn't one already
        if os.path.exists(paths.EVALUATION_RESULTS):
            model_results = pd.read_csv(paths.EVALUATION_RESULTS)
        else:
            model_results = pd.DataFrame([])

        # Here we perform an exhaustive evaluation for each of the models on each of the evaluation sets
        for model_name in model_names:
            logging.info('')

            for eval_name, eval_set in zip(eval_names, eval_sets):

                logging.info(f'Running {model_name} on "{eval_name}" evaluation data')

                result = evaluate_model(model_path=os.path.join(paths.TRAINED_MODELS, model_name), df=eval_set)

                result[0]['data_set'] = eval_name


                model_results = pd.concat([model_results, pd.DataFrame(result, index=[len(model_results)])])


        # Stash the results in a csv for later review
        model_results.to_csv(os.path.join(paths.EVALUATION_RESULTS))
    except OSError as ex:
        LOGGER.exception(f"Error in evaluation! Something is wrong with the directory: {ex}")
    except ValueError as ex:
        LOGGER.exception(f"Bad Model or data! {ex}")
    except Exception as ex:
        LOGGER.exception(f"Could not evaluate models: Exception - {ex}")



if __name__ == "__main__":
    main()
