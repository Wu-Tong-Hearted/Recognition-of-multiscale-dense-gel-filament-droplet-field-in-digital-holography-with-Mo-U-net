# Filament-droplet Field Detection Using Mo-U-net]{Recognition of multiscale dense gel filament-droplet field in digital holography with Mo-U-net
 # Overall of this project
## Structure of files
1. File Functions contains some support functions used in training including  callbacks (some train strategies), metrics, test metrics (metrics_in_novel) and show_img.

2. File data contains input img and ATM label. In this repository, we only give a small batch of data for valuation to decrease the file size.

3. File logs/fit contains the tracks of training that were recorded during training. You can use "tensorboard" to check it.

4. File model contains our Mo-U-net.

5. File result contains predictions saved during training and models (in this repository, we only give our trained model weights instead of the whole model to decrease the file size). Pred_avi is just a demo.

6. File test data contains part of our test EFI gained from five different experimental conditions.

7. File train_log contains the trained weights of Mo-U-net. If you want to valuate the performance or just employ Mo-U-net for another use, please first check that you have loaded the right weights (unoverfitted) into the Mo-U-net. In our study, the best epoch for weights should be lower than 10 epochs.
