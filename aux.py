#---------------------Auxiliary functions file------------------------
# 12/03/2025

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
from torchvision.transforms import ToPILImage
from import_drive import DRIVE
from skimage.morphology import medial_axis, area_closing
from PIL import Image
import sys
import os

sys.path.append("/home/joaolinaris/USP/IC/Projeto_retina/Segmentacao_vasos/UNET_DRIVE/DSE-skeleton-pruning")
from dsepruning import skel_pruning_DSE
# os.chdir("/home/jplinaris/UNET_DRIVE/")

def get_shift_tuples(value):
    
    radius = int(np.ceil(value))
    
    x_shifts, y_shifts = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    
    shifts = np.column_stack((x_shifts.ravel(), y_shifts.ravel()))  # Convert to list of (dx, dy) tuples
    # shifts = [(dx, dy) for dx, dy in shifts if (dx, dy) != (0, 0)]  # Returns without (0,0)
    return  shifts

def get_metrics(y_pred, y_true):
    """
    Expected input vectors:
    y_pred: raw output of the model (the probabilities)
    y_true: correct mask (just 0 or 255 values)
    BOTH STILL IN THE DEVICE (.cpu() already does the job of
    getting them into the cpu)
    """
    pred_copy = y_pred.clone()
    true_copy = y_true.clone()

    pred_copy = pred_copy.to(torch.device("cpu"))
    pred_copy = pred_copy.detach().numpy()
    pred_copy = pred_copy > 0.5
    pred_copy = pred_copy.astype(np.uint8)
    pred_copy = pred_copy.reshape(-1)

    true_copy = true_copy.to(torch.device("cpu"))
    true_copy = true_copy.detach().numpy()
    true_copy = true_copy > 0
    true_copy = true_copy.astype(np.uint8)
    true_copy = true_copy.reshape(-1)

    

    ac = accuracy_score(true_copy, pred_copy)
    f1 = f1_score(true_copy, pred_copy)
    rec = recall_score(true_copy, pred_copy)
    pre = precision_score(true_copy, pred_copy)

    del pred_copy, true_copy

    

    return [ac, f1, rec, pre]

def save_plot(y, title=None, y_label=None, curve_label=None, x_print_interval=20):
    """
    We suppose x axis is the number of epochs
    and that the metrics were collected every epoch.
    """
    epochs = len(y)
    # Originating the file name from the title
    if title != None:
        nome_arquivo = title.replace(" ", "_") 
        nome_arquivo += ".png" 
    else:
        nome_arquivo = "grafico_generico.png"
    
    # Plot construction

    plt.figure(figsize=(5,3))
    
    if curve_label != None:
        plt.plot(np.arange(1, epochs + 1), y, label=curve_label)
    else:
        plt.plot(np.arange(1, epochs + 1), y)
    
    plt.xlabel('Epochs')
    
    if y_label != None:
        plt.ylabel(y_label)
    
    plt.xticks(np.arange(1,epochs+1, x_print_interval))

    if title != None:
        plt.title(title)
    
    plt.legend()
    plt.savefig(nome_arquivo)

    return 0

def save_conj_plot(curve1, curve2, curve1_label=None, curve2_label=None, title=None, y_axis_label=None, x_print_interval=20):
    """
    We suppose x axis is the number of epochs
    and that the metrics were collected every epoch.
    """
    epochs = len(curve1)

    # Originating the file name from the title
    if title != None:
        nome_arquivo = title.replace(" ", "_") 
        nome_arquivo += ".png" 
    else:
        nome_arquivo = "grafico_generico.png"
    
    # Plot construction

    plt.figure(figsize=(5,3))
    
    if curve1_label != None:
        plt.plot(np.arange(1, epochs + 1), curve1, label=curve1_label)
    else:
        plt.plot(np.arange(1, epochs + 1), curve1)

    if curve2_label != None:
        plt.plot(np.arange(1, epochs + 1), curve2, label=curve2_label)
    else:
        plt.plot(np.arange(1, epochs + 1), curve2)
    
    
    plt.xlabel('Epochs')
    
    if y_axis_label != None:
        plt.ylabel(y_axis_label)
    
    plt.xticks(np.arange(1,epochs+1, x_print_interval))

    if title != None:
        plt.title(title)
    
    plt.legend()
    plt.savefig(nome_arquivo)

    return 0


def calculate_recall_straight(y_pred, y_true, ceil=2.0):
    """
    Will consider belonging to a straight vein
    if the distance to a black pixel is 
    less or equal to 'ceil'.

    Then it returns a vector:
    [recall, precision]
    """
    pred_copy = y_pred.clone().to(torch.device("cpu"))
    true_copy = y_true.clone().to(torch.device("cpu"))

    pred_copy = pred_copy.detach().numpy()
    pred_copy = pred_copy > 0.5
    pred_copy = pred_copy.astype(np.uint8)

    true_copy = true_copy.detach().numpy()
    true_copy = true_copy > 0
    true_copy = true_copy.astype(np.uint8)

    pred_copy = pred_copy[0] #[1,512,512] --> [512,512]
    true_copy = true_copy[0] #[1,512,512] --> [512,512]

    

    # Application of closing on the segmentation mask
    closed_true_copy = area_closing(true_copy)

    # Obtaining the skeleton
    skeleton_medial_axis, distances = medial_axis(closed_true_copy, return_distance=True)

    # Skeleton prunning
    skeleton_medial_axis = skel_pruning_DSE(skeleton_medial_axis, distances, np.ceil(distances.max()))

    # Compute the skeleton with the values of the distances
    dist_skel = np.where(skeleton_medial_axis>0, distances, 0) 

    # img = Image.fromarray((((dist_skel-dist_skel.min())/(dist_skel.max()-dist_skel.min()))*255).astype(np.uint8))
    # img.show()
    # exit(0)

    # Get unique values of dist_skel excluding 0 (values of the radius of vessels)
    values_dist_skel = np.unique(dist_skel)[1:] 

    all_values_dist_skel = values_dist_skel.copy()

    # Filtering values less than or equal to the ceiling considered (Only thin vessels remain)
    values_dist_skel = values_dist_skel[values_dist_skel <= ceil]

    #~~~~~~~~~~~~~~~~~~~~~~~Segmentation mask recriation with thin vessels only~~~~~~~~~~~~~~~~~~
    new_seg_true = np.zeros(dist_skel.shape)

    linhas = len(dist_skel)
    colunas = len(dist_skel[0])
    for value in values_dist_skel:
        shifts = get_shift_tuples(value)

        for i in range(linhas):
            for j in range(colunas):
                if abs(dist_skel[i][j] - value) <= 0.1:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            new_seg_true[i+dx][j+dy] = 255
    
        
    # img = Image.fromarray(new_seg_true.astype(np.uint8))
    # img.save(f"new_seg_true_before_reshaping.png") 

    # Filtering to get exactly the shape of the vessels, and not something rounded
    new_seg_true = np.where((true_copy>0) & (new_seg_true>0), 255, 0).astype(np.uint8)
    
    # # Saves the image of what is considered thin vessels
    # img = Image.fromarray(new_seg_true)
    # img.save("Straight_veins_seg_mask.png")
    # img = Image.fromarray((true_copy*255).astype(np.uint8))
    # img.save("Seg_mask.png")
    # img = Image.fromarray(new_seg_true.astype(np.uint8))
    # img.save("new_seg_true_before_adding_excluded_vessels.png")

    # exit(0)

    #~~~~~~~Thin vessels segmentation mask addition of lost vessels in prunning/closing process~~~~~~~~
    reconstructed_seg_mask = np.zeros(dist_skel.shape)

    linhas = len(dist_skel)
    colunas = len(dist_skel[0])
    
    for value in all_values_dist_skel:
        shifts = get_shift_tuples(value)

        for i in range(linhas):
            for j in range(colunas):
                if dist_skel[i][j] == value:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            reconstructed_seg_mask[i+dx][j+dy] = 255
    
    # Filtering to get exactly the shape of the vessels, and not something rounded
    reconstructed_seg_mask = np.where((true_copy>0) & (reconstructed_seg_mask>0), 255, 0).astype(np.uint8)

    # Gets exactly the excluded vessels
    excluded_vessels = np.where((true_copy>0) & (reconstructed_seg_mask==0), 255, 0).astype(np.uint8)

    # Concatenation of excluded_vessels seg mask with the thin vessels mask (we garantee they are small due to
    # their exclusion in the prunning/closing process)
    new_seg_true = np.where((new_seg_true>0) | (excluded_vessels>0), 255, 0).astype(np.uint8)

    # # Shows the difference between new_seg_true and true_copy (even if you consider the
    # # maximum vessel radius, it will be slightly different due to the applicaiton of area 
    # # closing in new_seg_true)
    # img_new_seg_true = np.expand_dims(new_seg_true, axis=-1)
    # img_new_seg_true = np.concatenate([img_new_seg_true, img_new_seg_true, img_new_seg_true], axis=-1)
    # img_seg_true = np.expand_dims(true_copy, axis=-1)
    # img_seg_true = np.concatenate([img_seg_true, img_seg_true, img_seg_true], axis=-1)
    
    # # Compares pixel to pixel if all channels are greater than 0
    # only_new_is_true = (img_new_seg_true > 0).all(axis=-1) & (img_seg_true == 0).all(axis=-1)
    # only_mask_is_true = (img_new_seg_true == 0).all(axis=-1) & (img_seg_true > 0).all(axis=-1)
    # both_true = (img_new_seg_true > 0).all(axis=-1) & (img_seg_true > 0).all(axis=-1)

    # # Inicializes the difference image
    # difference_img = np.zeros_like(img_new_seg_true, dtype=np.uint8)

    # # Applies colors to the interest points
    # difference_img[both_true] = [255, 255, 255]          # White
    # difference_img[only_new_is_true] = [219, 13, 2]      # Red
    # difference_img[only_mask_is_true] = [0, 255, 0]      # Green

    
    # img = Image.fromarray(difference_img.astype(np.uint8))
    # img.save("Difference_new_seg_and_seg_masks.png")
    # img = Image.fromarray(new_seg_true.astype(np.uint8))
    # img.save("new_seg_true_after_adding_excluded_vessels.png")


    # Calculates Recall
    tp = 0.0
    fn = 0.0
    for i in range(len(new_seg_true)):
        for j in range(len(new_seg_true[i])):
            if new_seg_true[i][j]:     
                if pred_copy[i][j]:
                    tp +=1
                else:
                    fn+=1
    # print(np.unique(new_seg_true))
    # print(np.unique(pred_copy))
    # print(f"Recall scikit learn: {recall_score((new_seg_true/255).astype(np.uint8).reshape(-1), pred_copy.reshape(-1))}")
    # print(f"Recall scikit learn: {recall_score((true_copy).astype(np.uint8).reshape(-1), pred_copy.reshape(-1))}")


    return tp/(tp+fn)

def calculate_precision_straight(y_pred, y_true, ceil=2.0):
    """
    Will consider belonging to a straight vein
    if the distance to a black pixel is 
    less or equal to 'ceil'.

    Then it returns a vector:
    [recall, precision]
    """
    pred_copy = y_pred.clone().to(torch.device("cpu"))
    true_copy = y_true.clone().to(torch.device("cpu"))

    pred_copy = pred_copy.detach().numpy()
    pred_copy = pred_copy > 0.5
    pred_copy = pred_copy.astype(np.uint8)

    true_copy = true_copy.detach().numpy()
    true_copy = true_copy > 0
    true_copy = true_copy.astype(np.uint8)

    pred_copy = pred_copy[0] #[1,512,512] --> [512,512]
    true_copy = true_copy[0] #[1,512,512] --> [512,512]

    # Application of closing on the segmentation mask
    closed_pred_copy = area_closing(pred_copy)

    # Obtaining the skeleton
    skeleton_medial_axis, distances = medial_axis(closed_pred_copy, return_distance=True)

    # Skeleton prunning
    skeleton_medial_axis = skel_pruning_DSE(skeleton_medial_axis, distances, np.ceil(distances.max()))
    
    # Compute the skeleton with the values of the distances
    dist_skel = np.where(skeleton_medial_axis>0, distances, 0) # Esqueleto com as distâncias

    # Get unique values of dist_skel excluding 0 (values of the radius of vessels)
    values_dist_skel = np.unique(dist_skel)[1:] # Valores únicos de dist_skel (raios dos vasos) (tirando o 0)
    
    all_values_dist_skel = values_dist_skel.copy()

    # Filtering values less than or equal to the ceiling considered (leaves thin vessels only)
    values_dist_skel = values_dist_skel[values_dist_skel <= ceil]

    #~~~~~~~~~~~~~~~~~~~~~~~Segmentation mask recriation with thin vessels only~~~~~~~~~~~~~~~~~~  
    new_seg_pred = np.zeros(dist_skel.shape)

    linhas = len(dist_skel)
    colunas = len(dist_skel[0])
    
    for value in values_dist_skel:
        shifts = get_shift_tuples(value)

        for i in range(linhas):
            for j in range(colunas):
                if dist_skel[i][j] == value:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            new_seg_pred[i+dx][j+dy] = 255
    
    # Filtering to get exactly the shape of the vessels, and not something rounded
    new_seg_pred = np.where((pred_copy>0) & (new_seg_pred>0), 255, 0).astype(np.uint8)
    
    # # Saves the image of what is considered thin vessels
    # img = Image.fromarray(new_seg_pred)
    # img.save("Straight_veins_pred.png")

    #~~~~~~~Thin vessels segmentation mask addition of lost vessels in prunning/closing process~~~~~~~~
    reconstructed_pred_seg_mask = np.zeros(dist_skel.shape)

    linhas = len(dist_skel)
    colunas = len(dist_skel[0])
    
    for value in all_values_dist_skel:
        shifts = get_shift_tuples(value)

        for i in range(linhas):
            for j in range(colunas):
                if dist_skel[i][j] == value:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            reconstructed_pred_seg_mask[i+dx][j+dy] = 255
    
    # Filtering to get exactly the shape of the vessels, and not something rounded
    reconstructed_pred_seg_mask = np.where((pred_copy>0) & (reconstructed_pred_seg_mask>0), 255, 0).astype(np.uint8)

    # Gets exactly the excluded vessels
    excluded_vessels = np.where((pred_copy>0) & (reconstructed_pred_seg_mask==0), 255, 0).astype(np.uint8)

    # Concatenation of excluded_vessels seg mask with the thin vessels mask (we garantee they are small due to
    # their exclusion in the prunning/closing process)
    new_seg_pred = np.where((new_seg_pred>0) | (excluded_vessels>0), 255, 0).astype(np.uint8)

    # Calculates Precision
    tp = 0.0
    fp = 0.0
    for i in range(len(new_seg_pred)):
        for j in range(len(new_seg_pred[i])):
            if new_seg_pred[i][j]:     
                if true_copy[i][j]:
                    tp +=1
                else:
                    fp+=1

    return tp/(tp+fp)


def main():

    root_path = "/home/joaolinaris/USP/IC/Projeto_retina/Datasets/FULL_DRIVE/"   
    test_data = DRIVE(root_path, test_data=True, augmentation=False, full_drive=True)
    img = test_data[5][0]
    seg = test_data[5][1]
    model = torch.load("/home/joaolinaris/USP/IC/Projeto_retina/Segmentacao_vasos/UNET_DRIVE/MODELS/UNet_bce_w1.pth", weights_only=False)
    model.eval()
    dif = 0
    with torch.no_grad():
        # print(train_data_no_aug[0][0].shape)
        for img, seg, files in test_data:
            img = img.unsqueeze(0)
            seg = seg.unsqueeze(0)
            pred = model(img)
            # print(pred.shape)
            # print(img.shape)
            # print(seg.shape)
            # print(torch.unique(seg))
            pred = pred[0] # Pegando a única imagem do batch retornado (seg tem shape [1,512,512])
            seg = seg[0]

            # # Uncomment to see that the model is unsure what exactly is considered a thin vessel
            # pred = np.where(pred>0.5, 255, 0)
            # print(pred.shape)
            # pred_img = (pred.numpy()*255).astype(np.uint8)
            # pred_img = ((pred > 0.5).numpy() * 255).astype(np.uint8)
            # pred_img = pred_img[0]
            # print(pred_img.shape)
            # pred_img = Image.fromarray(pred_img)
            # # pred_img.save("Weights0_prediction_train_0.png")
            # pred_img.show()
            # seg_img = (seg.numpy()*255).astype(np.uint8)
            # seg_img = seg_img[0]
            # print(seg_img.shape)
            # seg_img = Image.fromarray(seg_img)
            # # pred_img.save("Weights0_prediction_train_0.png")
            # seg_img.show()
            metrics = get_metrics(pred, seg)
            rec_straight = calculate_recall_straight(pred, seg, ceil=10)
            
            dif += abs(rec_straight-metrics[2])
            # print(metrics)
            # print(rec_straight)
            # print(prec_straight)
        print(f"Diferença: {dif}")

        

    return 0


if __name__ == "__main__":
    main()








