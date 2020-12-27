import xlsxwriter
import os
import numpy

#PATH = '/Users/cesaremagnetti/projects/PyCharm/KURF/CesareMagnetti/results/accuracies&plots/accuracies/all_dataset/RAW/'
PATH = '/Users/cesaremagnetti/projects/PyCharm/KURF/CesareMagnetti/results/accuracies&plots/accuracies/small_dataset/webcam/'

#filenames = ["resnet18_accuracy_5_52_raw.txt", "vgg13_1_44_accuracies_raw.txt", "resnet18_aug_accuracy_1_90_raw.txt", "vgg13_aug_2_30_accuracies_raw.txt"]
filenames = ['resnet18_accuracy_5_52_cam.txt', 'vgg13_1_44_accuracies_cam.txt', 'resnet18_aug_accuracy_1_90_cam.txt', 'vgg13_aug_2_30_accuracies_cam.txt' ]
for f in filenames:
    k=0
    results = []
    new = []
    NAME = PATH+f
    assert os.path.exists(NAME), "error, file does not exist!"
    file = open(NAME, "r")
    for line in file:
        if k%8 == 0 and k!=0 and line.strip():
            new = [float(i) for i in new]
            new = [float("{0:.2f}".format(i)) for i in new]
            new.append(sum(new)/8)
            results.append(new)
            new = []

        temp = line.split(":")
        if len(temp) == 2:
            new.append(temp[1].strip().strip('%'))
            k+=1
    file.close()

    results = numpy.array(results).astype(float)
    name = f.split('.')[0] + '.xlsx'
    workbook = xlsxwriter.Workbook(name)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    for Epoch,Abdomen,Background,Head,Limbs,Placenta,Spine,Thorax,Average in results:
        worksheet.write(row, col, Epoch)
        worksheet.write(row, col+1, Abdomen)
        worksheet.write(row, col+2, Background)
        worksheet.write(row, col+3, Head)
        worksheet.write(row, col+4, Limbs)
        worksheet.write(row, col+5, Placenta)
        worksheet.write(row, col+6, Spine)
        worksheet.write(row, col+7, Thorax)
        worksheet.write(row, col + 8, Average)
        row +=1

    workbook.close()




