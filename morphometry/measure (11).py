import os
import PIL.ImageDraw as ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image, ImageFont
from skimage.draw import polygon
from skimage import measure
from pathlib import Path
from skimage.draw import ellipse
from skimage.transform import rotate
from collections import Counter
import configparser
import argparse


def morphometry(guard_value, pore_value, mask_crop):
    # Extract morphometry
    g_w = g_w1 = g_h = p_w = p_h = gch1 = gch2 = 0

    # Get center values
    m_h, m_w = mask_crop.shape

    # Get the pore
    pore = np.array(np.where(mask_crop[:,:] == pore_value))
    #print(mask_crop)
    # Iterate over range to remove artifacts in data
    for i in range(-3, 3):
        # Guard cell measurements
        #print(mask_crop)
        if int(m_w / 2) + i >= m_w or int(m_h / 2) + i >= m_h: continue
        guard_height = np.where(mask_crop[:, int(m_w / 2) + i] == guard_value)
        guard_width = np.where(mask_crop[int(m_h / 2)  + i,:] == guard_value)

        
        # Calculate guard width and height
        temp_g_w = sum(len(i) for i in guard_width)
        if temp_g_w > g_w:
            g_w = temp_g_w
        temp_g_h = sum(len(i) for i in guard_height)
        if temp_g_h > g_h:
            g_h = temp_g_h

        # Get maximum value of the guard cell width
        temp_gch1 = (len(guard_height[0]))
        if temp_gch1 > gch1:
            gch1 = temp_gch1

        # Pore measurements
        pore_height = np.where(mask_crop[:, int(m_w / 2) + i] == pore_value)
        pore_width = np.where(mask_crop[int(m_h / 2) + i,:] == pore_value)

        # Pore width and height
        temp_p_h = (len(pore_height[0]))
        if temp_p_h > p_h:
            p_h = temp_p_h
        temp_p_w = (len(pore_width[0]))
        if temp_p_w > p_w:
            p_w = temp_p_w

    # Remove any error or noise within data
    #if gch2 == 0 or gch1 == 0:
    if gch1 == 0 and p_h==0:
        return None, None, None, None
    # Total width and height
    s_w = p_w + g_w
    s_h = p_h + gch1 + gch2

    # Output to csv file
    PL = (p_w * um_per_pixel)
    GCW_1 = (gch1 * um_per_pixel)
    GCW_2 = (g_w * um_per_pixel)
    PSG_W = (s_h * um_per_pixel)
    
    return (PL, GCW_1, GCW_2, PSG_W)

def split_set(arr, space):
    indices = [i + 1 for (x, y, i) in zip(arr, arr[1:], range(len(arr))) if space < abs(x - y)]
    result = [arr[start:end] for start, end in zip([0] + indices, indices + [len(arr)])]
    return result

def contour_area(contour):
    a = 0
    prev = contour[-1]
    for pt in contour:
        #pt = pt * um_per_pixel
        a += (prev[0] * pt[1] - prev[1] * pt[0])
        prev = pt
    a *= 0.5
    return abs(a)

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = 0.5 * abs(area)
    return area


def bounding_box(points, padding, w, h):
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)

    if min_x <= padding: min_x = 0
    else: min_x = min_x - padding
    if min_y <= padding: min_y = 0
    else: min_y = min_y - padding
    if max_x + padding >= w: max_x = h
    else: max_x = max_x + padding
    if max_y + padding >= w: max_y = w
    else: max_y = max_y + padding

    return [int(min_x), int(min_y), int(max_x), int(max_y)]

if __name__ == "__main__":
    # Variables
    working_dir = "/content/PollenCounter"
    image_dir = "/predict/unannotated/"
    predict_dir ="/predict/annotated/"


    # Poplar  34, 120
    # Wheat gc 23, pore 150
    pore_value = 255
    guard_value = 34
    padding = 2
    artifact_padding = 15
    min_contour = 100
    um_per_pixel = 1.2547
    #poplar 0.181818 wheat  0.12547
    save_individual = False
    density_value = 63
    #68 wheat 277 poplar
    # Empty excel output and declare headers
    excel_output = []
    default_length = 50.0
    guess_elevation = False
    columns = "ID, Length, Width"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Specify config file")
    args = parser.parse_args()

    # Check to see if config file has been specified
    if args.config:
      # Parse config file
      try:
          config = configparser.ConfigParser()
          config.read(args.config)
          working_dir = config['measure_directories']['working_dir']
          original_lengths = config['elevations']
          truth_string = config['elevations']['guess_elevation'].strip(" ")
          if truth_string == "True":
            guess_elevation = True
            print("Guess elevation is on")
          else:
            guess_elevation = False
            print("Guess elevation is off")
          um_per_pixel = float(config['measure_directories']['um_per_pixel'])
          default_length =  float(config['elevations']['default_length'])

      except:
          print("==> Error reading config file")
          exit()
    #print('')
    # Iterate through files
    files = list(Path(working_dir + image_dir).glob('*.jpg'))
    for file in files:
        path = Path(file.name)
        filename = path.stem
        if "_stats" in file.name: continue
        if "psme" in file.name:
          elevation = filename.split("psme-",1)[1]
          elevation = elevation.split("-",1)[0]
        elif "PSME" in file.name:
          elevation = filename.split("PSME_",1)[1]
          elevation = elevation.split("_",1)[0]
        else: 
          elevation = "-1"
        
        elevation = "length_"+elevation
        if (guess_elevation==True) and (elevation in original_lengths.keys()):
          original_length = float(original_lengths.get(elevation))
        else: 
          original_length = default_length
          elevation = "default_length"

      
          
        # Image variables for averages
        avg_pore_length = avg_gcw_1 = avg_gcw_2 = avg_psg =  avg_pore_area = avg_gc_area = avg_gsmax_c = avg_gsmax_e = avg_operational_gs = 0
        # Check if mask exists for file
        file_name = str(file.stem)

        # Output to excel
        excel_output.append(file_name + ", " + columns)

        # Open images
        print(file_name)     
        print("elevation config: " + str(elevation))
        print("original length: " + str(original_length))
        original_img = Image.open(file).convert('RGB')
        mask_img = Image.open(working_dir + predict_dir + file_name+"_mask.png").convert('L')

        # Initialise draw
        draw = ImageDraw.Draw(original_img)
        font = ImageFont.load_default()

        # Convert PIL images to numpy arrays
        mask_np = np.array(mask_img)

        # Size
        w, h = original_img.size
        #print(mask_np)
        # Find contours
        graph = np.where(mask_np == guard_value, 0, mask_np)
        edited_contour12 = np.where(mask_np == pore_value, 255, mask_np)
        edited_contour34 = np.where(mask_np == guard_value, 255, mask_np)
        contours12 = measure.find_contours(np.where(mask_np == guard_value, 0, edited_contour12), level=61)
        contours34 = measure.find_contours(np.where(mask_np == pore_value, 0, edited_contour34), level=61)



        # Select the largest contiguous contour
        contours12 = sorted(contours12, key=lambda x: len(x), reverse=True)
        contours34 = sorted(contours34, key=lambda x: len(x), reverse=True)
        # Variables
        id, total_count, total_pore, total_guard = 0, None, 0, 0
        # Iterate through each of the contours (stomata)
        count1 = 0
        count2 = 0
        count34 = 0 
        pollen_class = 0
        class_list = [contours12,contours34]
        for contours in class_list:
          for contour in contours:
              # If contour is too small: ignore
              if len(contour) < min_contour:
                  continue
              if len(contour) > 10000:
                  continue

              min_x, min_y, max_x, max_y = bounding_box(contour, padding, w, h)

              # Check if border, and skip
              #if min_x == 0 or min_y == 0 or max_x == w or max_y == h: continue

              # Increment counts
              id += 1

              # Crop the images
              mask_crop = mask_np[min_x:max_x,min_y:max_y]
              # Get coordinates inside poly
              xx, yy = polygon(contour[:, 0], contour[:, 1], mask_np.shape)

              # Calculate eigenvalue and vectors
              x = xx - np.mean(xx)
              y = yy - np.mean(yy)
              coords = np.vstack([x, y])
              cov = np.cov(coords)
              evals, evecs = np.linalg.eig(cov)
              sort_indices = np.argsort(evals)[::-1]

              # Eigenvector
              x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue

              # Get angle of rotation
              theta = np.arctan((x_v1)/(y_v1))

              # Convert mask to 3 channels
              mask_multichannel = np.stack((mask_crop,)*3, axis=-1)

              # Rotate the mask and save
              mask = Image.fromarray(mask_multichannel)
              mask_aligned = mask.rotate(57.2958 * theta, expand=True, fillcolor=(255,255,255))

              # Begin counting
              unique, counts = np.unique(mask_crop, return_counts=True)
              local_counts = dict(zip(unique, counts))
              if total_count == None:
                  total_count = local_counts
              else:
                  total_count = Counter(total_count) + Counter(local_counts)

              # Extract morphometry
              mask_crop = np.asarray(mask_aligned)
              mask_crop = mask_crop[:,:,0]
              if contours==contours12: 
                PL, GCW_1, GCW_2, PSG_W = morphometry(guard_value, pore_value, mask_crop)
              else:
                PL, GCW_1, GCW_2, PSG_W = morphometry(pore_value, guard_value, mask_crop)
                #print('hey')
              if GCW_1 == None:
                  id -= 1
                  continue
              # New area estimation for pore
              pore_crop = np.where(mask_crop == guard_value, 255, mask_crop)
              pore_crop = np.where(pore_crop == pore_value, 0, 255)
              pore_contour = measure.find_contours(pore_crop, level=1, fully_connected='high', positive_orientation='high')

              '''for p in pore_contour[0]:
                  pore_crop[int(p[0]), int(p[1])] = 5

              # Save mask image
              mask = Image.fromarray(pore_crop).convert('RGB')
              mask.save(working_dir + predict_dir + file_name + "_testing2.jpg", subsampling=0, quality=100)'''


              pore_area = 0
              # Total Area
              unique, counts = np.unique(mask_crop, return_counts=True)
              total_counts = (dict(zip(unique, counts)))


              # Individual areas
              guard_area = total_counts.get(guard_value, 0.0) * um_per_pixel
              pore_area = total_counts.get(pore_value, 0.0) * um_per_pixel
              pore_area = pore_area * um_per_pixel
              # Expand numpy dimensions
              #c = np.expand_dims(contour[0].astype(np.float32), 1)
              # Convert it to UMat object
              #c = cv2.UMat(c)
              #area2 = cv2.contourArea(c)
              #area2 = 0

              # Total areas
              total_pore += pore_area
              total_guard += guard_area


              # Append values to averages
              #avg_pore_length += PL
              avg_pore_length = 0
              avg_gcw_1 += GCW_1
              #avg_gcw_1= 0
              #avg_gcw_2 += GCW_2
              avg_gcw_2 += GCW_2
              #avg_psg += PSG_W
              avg_psg = 0 
              #avg_pore_area += pore_area
              avg_pore_area =0
              #avg_gc_area += guard_area
              avg_gc_area = 0
              #avg_gsmax_c += gsmax_circular_mol
              avg_gsmax_c = 0
              avg_gsmax_e =0
              #avg_gsmax_e += gsmax_ellipitical_mol
              #avg_operational_gs += operational_gs
              avg_operational_gs = 0


              # Visualise on image
              if contours==contours12:
                if(GCW_2>2*original_length):
                  pollen_class = 1
                  count1 += 1
                  color = (0,255,255)
                elif(GCW_2>original_length-artifact_padding):
                  pollen_class = 2
                  count2 += 1
                  color = (255,0,0)
                else: 
                  pollen_class = 3
                  count34 += 1
                  color = (255,255,0)
              else:
                pollen_class = 3
                count34 += 1
                color = (255,255,0)
              draw.text((max_y - 50,min_x), "Class: " + str(pollen_class), fill=color, font=font)
              excel_output.append("," + str(id) + ", " + str(GCW_2) + ", " + str(GCW_1) + ", " +  str(pollen_class))
              draw.text((max_y - 50,min_x - 10), "ID: " + str(id), fill=color, font=font)
              #draw.text((max_y - 50,min_x - 10), "Width: " + "{:.3f}".format(GCW_1), fill=(0, 0, 0), font=font)
              draw.text((max_y - 50,min_x - 20), "Length: " + "{:.3f}".format(GCW_2), fill=(0, 0, 0), font=font)



              # Save individual crop of stomata
              #if save_individual:
              if False:
                  # Draw center line over mask
                  temp = np.zeros(mask_crop.shape)
                  temp = np.copy(mask_crop)

                  # Save mask image
                  mask = Image.fromarray(temp).convert('RGB')
                  mask.save(working_dir + predict_dir + file_name + "_mask_"  + str(id) + '_sample.png', subsampling=0, quality=100)
          # Check if empty image


          if id == 0: continue
          # Excel output averages in columns: Length of Pore, GCW_1, GCW_2, PSG(W), Pore Area, GC Area

          # Density of Image
          density = id / ((w*um_per_pixel/1000)*(h*um_per_pixel/1000))

          # Outputs to display on original image
        draw.text((10, 10), "Pollen Count:     " + str(id), fill=(0, 0, 0), font=font)
        draw.text((10, 20), "Class 1 Count:     " + str(count1), fill=(0, 0, 0), font=font)
        draw.text((10, 30), "Class 2 Count:     " + str(count2), fill=(0, 0, 0), font=font)
        draw.text((10, 40), "Class 3/4 Count:     " + str(count34), fill=(0, 0, 0), font=font)
          #draw.text((10, 30), "Stomata Density:   " + str(float("{:.2f}".format(density))), fill=(0, 0, 0), font=font)
        excel_output.append(",Average, " + str(avg_gcw_2 / id) + ", " + str(avg_gcw_1 / id))
        excel_output.append(",Class 1 Count, " + str(count1))
        excel_output.append(",Class 2 Count, " + str(count2))
        excel_output.append(",Class 3/4 Count, " + str(count34))





          # Save original image with contours and stats
        original_img = Image.blend(original_img, Image.fromarray(mask_np).convert('RGB'), 0.3)
        original_img.save(working_dir + predict_dir + file_name + "_stats.png", subsampling=0, quality=100)

    # Save the csv file
    with open(working_dir + predict_dir + 'morphometry.csv','w') as file:
        for l in excel_output:
            file.write(l)
            file.write('\n')
