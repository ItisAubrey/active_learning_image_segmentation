import os

from data_processing_utils import load_data, saveList, openList
from logger_utils import get_logger
from active_learning_utils import scanPool, saveSampleResult
from model_evaluation_utils import evaluation, evaluationTest
from model_utils import crossVali
import pandas as pd
import numpy as np
logger = get_logger(__name__)



def al_segmentation():
    method = os.getenv('METHOD', '')
    # Define dataset path and load the data
    dataset_path = "/dataset_to_use/ProData/"
    (train_x, train_y, train_z), (pool_x, pool_y, pool_z), (test_x, test_y, test_z) = load_data(dataset_path,
                                                                                                split2=0.2, split3=0.05,
                                                                                                augRatio=0)

    # Log dataset sizes
    logger.info(f"Train: {len(train_x)} - {len(train_y)} - {len(train_x)}")
    logger.info(f"Pool: {len(pool_x)} - {len(pool_y)} - {len(pool_z)}")
    logger.info(f"Test: {len(test_x)} - {len(test_y)} - {len(test_z)}")

    # Save the data paths to text files for future use
    saveList("multiInitialSet/train_xm.txt", train_x)
    saveList("multiInitialSet/train_ym.txt", train_y)
    saveList("multiInitialSet/train_zm.txt", train_z)
    saveList("multiInitialSet/pool_xm.txt", pool_x)
    saveList("multiInitialSet/pool_ym.txt", pool_y)
    saveList("multiInitialSet/pool_zm.txt", pool_z)
    saveList("multiInitialSet/test_xm.txt", test_x)
    saveList("multiInitialSet/test_ym.txt", test_y)
    saveList("multiInitialSet/test_zm.txt", test_z)

    # Initialize a summary DataFrame to store results
    summary = pd.DataFrame(columns=["method", "iou_value", "dic_coef_value", "dice_loss_value", "f1_value"])
    selectedRecord = []
    kfold = 3  # Number of folds for cross-validation
    method =method  # Set method for active learning
    logger.info("======current method:", method)

    # Reopen the data files again for use in the loop
    train_x = openList("multiInitialSet/train_xm.txt")
    train_y = openList("multiInitialSet/train_ym.txt")
    train_z = openList("multiInitialSet/train_zm.txt")
    pool_x = openList("multiInitialSet/pool_xm.txt")
    pool_y = openList("multiInitialSet/pool_ym.txt")
    pool_z = openList("multiInitialSet/pool_zm.txt")
    test_x = openList("multiInitialSet/test_xm.txt")
    test_y = openList("multiInitialSet/test_ym.txt")
    test_z = openList("multiInitialSet/test_zm.txt")

    # Iterate for 20 active learning cycles
    for i in range(20):
        logger.info(f"===current iteration: {i}")

        # Perform cross-validation
        crossVali(kfold, 100)
        SCORE = []

        # Log and perform evaluation for the current cycle
        logger.info("Evaluating the current model...")
        evaluation(SCORE, kfold)

        # Compute the average score for the fold evaluations
        pureScore = [s[1:] for s in SCORE]
        pureScore = np.mean(pureScore, axis=0)
        logger.info(f"Evaluation result for method {method}: {pureScore}")

        # Convert the score to a list and add method as the first element
        pureScore = pureScore.tolist()
        pureScore.insert(0, method)

        # Append the score to the summary DataFrame
        summary.loc[len(summary.index)] = pureScore

        # Select samples from the pool based on uncertainty
        selected = []
        logger.info("Scanning pool for uncertain samples...")
        selected = scanPool(selected, method, kfold)
        logger.info("The 1st and 2nd most uncertain samples are:")
        logger.info(selected[0])
        logger.info(selected[1])

        # Add selected samples to the training set and remove from the pool
        selectedList = [method]
        for j in range(6):
            cur = selected[j]
            pool_x.remove(cur[0])
            pool_y.remove(cur[1])
            pool_z.remove(cur[2])
            train_x.append(cur[0])
            train_y.append(cur[1])
            train_z.append(cur[2])
            name = cur[0].split("/")[-1]
            selectedList.append(name)

        # Record the selected samples
        selectedRecord.append(selectedList)

        # Log and save sample results for the current iteration
        logger.info("Saving sample results for this iteration...")
        saveSampleResult(test_x[10], test_y[10], test_z[10], method, i, kfold)
        saveSampleResult(test_x[5], test_y[5], test_z[5], method, i, kfold)

    # After completing the active learning cycles, log and save the summary
    logger.info("Final evaluation summary:")
    print(summary)
    summary.to_csv("summaryMulti.csv")

    # Save the selected samples for future analysis
    logger.info("Saving selected samples record...")
    print(selectedRecord)
    np.savetxt("selectedRecordmulti.csv", selectedRecord, delimiter=", ", fmt="%s")

    # Perform additional evaluation on the test set after active learning
    evaluationScore = []
    evaluationTest(evaluationScore, 3)

    # Convert the evaluation results to a DataFrame for easier analysis
    ScoreSeperateList = pd.DataFrame(evaluationScore, columns=[
        'ImageName', 'iou_value1', 'iou_value2', 'dic_coef_value1', 'dic_coef_value2',
        'dice_loss_value1', 'dice_loss_value2', 'f1_value1', 'f1_value2'
    ])

    # Convert any TensorFlow tensors to scalar values for consistency
    ScoreSeperateList['iou_value1'] = ScoreSeperateList['iou_value1'].apply(lambda x: x.numpy().item())
    ScoreSeperateList['iou_value2'] = ScoreSeperateList['iou_value2'].apply(lambda x: x.numpy().item())
    ScoreSeperateList['dic_coef_value1'] = ScoreSeperateList['dic_coef_value1'].apply(lambda x: x.numpy().item())
    ScoreSeperateList['dic_coef_value2'] = ScoreSeperateList['dic_coef_value2'].apply(lambda x: x.numpy().item())
    ScoreSeperateList['dice_loss_value1'] = ScoreSeperateList['dice_loss_value1'].apply(lambda x: x.numpy().item())
    ScoreSeperateList['dice_loss_value2'] = ScoreSeperateList['dice_loss_value2'].apply(lambda x: x.numpy().item())

    # Save the evaluation results to a CSV file for later use
    ScoreSeperateList.to_csv("MultiScoreSeperate.csv")