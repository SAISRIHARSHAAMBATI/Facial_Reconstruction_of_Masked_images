# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:09:06 2021

@author: sheetal
"""

import matplotlib.pyplot as plt

from initialization_pred import *
###############################################################################
# Local Functions
###############################################################################


###############################################################################

###############################################################################
# Start Main Code
###############################################################################




# predicting x

          
###############################################################3


###################################################################


####################
def create_file(dataset_name ,dataset):
    file_name = CNFG.DATASET_POINTS + '\\'+ dataset_name +'.py'
    #print(' writing file - ', file_name)
    f = open(file_name, 'w+')  # 
    f.write(dataset_name)
    f.write(' = ' )
    
    #print(dataset_name, str(len(dataset_name)))
    
    f.write(str((dataset)))
    f.close()
    
########################    



####################
def create_file1(dataset_name ,dataset):
        file_name = CNFG.DATASET_POINTS + '\\'+ dataset_name +'.py'
        #print(' writing file - ', file_name)
        f = open(file_name, 'w+')  # 
        f.write(dataset_name)
        f.write(' = [' )
        

        col = len(dataset[0])

        #for in range(0 , len(dataset_name))
        for row in range (0, len(dataset)):
            f.write(' [' )
            for colm in range(0, col):
                f.write( str(dataset[row][colm]) )
                if len(dataset[0]) ==1:
                    pass
                if colm > 0 and colm < len(col):
                   f.write(' ,' ) 
            f.write('],' )   
        #f.write(str((dataset)))
        f.close()
            
########################
###############################################################################


def main(img_analyse_count = "all", show_anim = False, 
         save_file = False, black_bg = False):
    

    # folder containing original images
    original_images = CNFG.IMAGES_FOLDER + "\\" + CNFG.ORIGINAL_IMAGES_FOLDER
    
    # list of images inside the original image folder
    original_images = glob.glob(original_images+"\\*.png")
    original_images = (natsort.natsorted(original_images))
    
    if not img_analyse_count == 'all':
        temp_imgs = []
        for count in range(0,img_analyse_count):
           temp_imgs.append(original_images[count])

        original_images = temp_imgs
    #
    print('total images = ', str(len(original_images)))
    
    
    x_train = np.asarray(x_vis.x_vis_normalized) # dim x m
    x_test = np.asarray(x_hid.x_hid_normalized) # 1 x m
    print('x_train_len = ', str(len(x_train)))
    print('x_test_len = ', str(len(x_test)))
    # print('x_train = ', str((x_train)))
    # print('x_test = ', str((x_test)))
    
    
    y_train = np.asarray(y_vis.y_vis_normalized) # dim x m
    y_test = np.asarray(y_hid.y_hid_normalized) # 1 x m
    print('y_train_len = ', str(len(y_train)))
    print('y_test_len = ', str(len(y_test)))
    
    # print('y_train = ', str((y_train)))
    # print('y_test = ', str((y_test)))
    
    
    
    num_top_points = len(x_train)
    num_bot_points = len(x_test)
    
    
    
    
    threshold = 0.5
    
    layer_0 = num_top_points
    layer_1 = num_bot_points
    
    costs = [] # to plot graph 
    cost_x = []
    cost_y = []
    
    hist_x = []
    hist_y = []
    
    
    EPOCHS = 501
    x_epochs = 500 #EPOCHS # 4001
    y_epochs = 500 #EPOCHS # 7001
    
    width = 500
    height = 500
       
    ################################################

    
    # Making x predictions
    print('\n Predicting masked x points \n')
    layers_x = [NN.Layer(layer_0, layer_1, 'sigmoid')]
    pred_x,_,_ = NN.predict_points(layers_x, x_train, x_test, cost_x, hist_x,x_epochs,'x ')
    
    # Making y predictions
    print('\n Predicting masked y points \n')
    layers_y = [NN.Layer(layer_0, layer_1, 'sigmoid'),]
    pred_y,_,_  = NN.predict_points(layers_y, y_train,y_test, cost_y, hist_y, y_epochs,'y ')
    
    
    
    pred_hid_x_scaled = (pred_x * x_hid_scale_fact * hid_width_mat).astype(int)
    print('length pred_hid_x_scaled  = ', str(len(pred_hid_x_scaled)))
    
    
    x_error =  np.sum (abs(hid_x_scaled - pred_hid_x_scaled)) / len(hid_x_scaled) 
    #print(x_error)
    
    pred_hid_y_scaled = (pred_y * y_hid_scale_fact * hid_width_mat).astype(int)
    y_error =  np.sum (abs(hid_y_scaled - pred_hid_y_scaled)) / len(hid_y_scaled)
    
    # print(' vis_x = \n', str((act_x_scaled)))
    # print('actual_hid x = \n', str(hid_x_scaled))
    # # print(' ')
    # print('Prediction x = \n', str(pred_hid_x_scaled) )
    #x_err = abs(hid_x_scaled - pred_hid_x_scaled)
    #print('Error in x  = ', str(x_err))
    
    
    # print(' vis_y = \n', str((act_y_scaled)))
    # print('actual_hid y = \n', str(hid_y_scaled))
    # print(' ')
    # print('Prediction y = \n', str(pred_hid_y_scaled) )
    # print('Error in y  = ', str(y_error))
    
    
    
    print("Error in prediction of the hidden points:")
    print("Error in predicting 'x' coord. = ", str(x_error), str( ''))
    print("Error in predicting 'y' coord. = ", str(y_error), str( ''))
    
    # print('error y  = ', str(y_error), str( ''))
    
    plt.figure(1)
    plt.plot(cost_x )
    plt.xlabel('Epochs')
    plt.ylabel('Error- X')
    plt.show()
    
    
    plt.figure(2)
    plt.plot(cost_y)
    plt.xlabel('Epochs')
    plt.ylabel('Error - Y')
    plt.show()

    ########################
    
    #print('length pred_hid_x_scaled 2 = ', str((pred_hid_x_scaled)))
    # save scaled predicted x and y
    create_file('pred_hid_x_scaled', pred_hid_x_scaled )
    create_file('pred_hid_y_scaled', pred_hid_y_scaled )

    # save cost of predicted x and y
    create_file('cost_x', cost_x )
    
    create_file('cost_y', cost_y )
    
    # save hist of predicted x and y
    create_file('hist_x', hist_x )
    create_file('hist_y', hist_y )
    
    face_top_points_x    = act_x_scaled
    face_top_points_y    = act_y_scaled
    
    face_bottom_points_x = hid_x_scaled
    face_bottom_points_y = hid_y_scaled
    ########################

    img_count = 0
    if show_anim:
    
        for images in original_images:
            print("\n Showing prediction result for ", images)
            ANIM.display_predictions(images, img_count,hist_x,hist_y,
                                    x_hid_scale_fact, y_hid_scale_fact,
                                    hid_width_mat,hid_height_mat,
                                    face_top_points_x,face_top_points_y,
                                    face_bottom_points_x,face_bottom_points_y,
                                    save_file, black_bg)
            img_count = img_count + 1
        
        #
        #cv2.waitKey(0)
    #

############################################

if __name__ == "__main__":    
    # call the main function
    main()

# 
###############################################################################
# End Main Code
###############################################################################
