import matplotlib.pyplot as plt
import time

from utils import *
import os.path as osp
from sklearn.metrics import pairwise_distances

# Define ONE training epoch
def train_step(model, train_data_loader, optimizer, device):
   # Step 1: set model to training mode
   model.train()
   # Step 2: Iterate through batches returned by the train_data_loader
   total_loss = 0.0
   for batch_i, data in enumerate(train_data_loader):
      # Step 3: put data to device. eg. 'cuda'
      data = data.to(device)
      # Step 4: zero out gradient
      optimizer.zero_grad()
      # Run data through model
      out = model(data)

      # Step 5: Calculate loss: Mean Absolute Error
      loss = (out.squeeze() - data.y).abs().mean()
      # Step 6: Back propogation
      loss.backward()

      # Step 7: adjust the parameters by the gradients collected in the backward pass
      optimizer.step()
      #Step 8: update total_loss
      total_loss += loss.item() * data.num_graphs

   # Step 8: calculate step loss
   loss = total_loss / len(train_data_loader.dataset)

   return model, loss

# Define the testing/validating step
@torch.no_grad()
def test(model, data_loader, device):
   # Step 1: Sett model to testing mode
   model.eval()
   total_error = 0

   """
   Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode.
   Also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
   """
   my_out = torch.empty(0).to(device)
   my_y = torch.empty(0).to(device)
   # my_f = torch.empty(0).to(device)

   with torch.no_grad():
      for i, data in enumerate(data_loader):
         #for data in data_loader:
         # Step 2: put data to device   
         data = data.to(device)
         # Step 3: run the data through model
         out = model(data)
         # Step 4: update error
         total_error += (out.squeeze() - data.y).abs().sum().item()
       
         my_out = torch.cat((my_out, out), 0)
         my_y = torch.cat((my_y, data.y), 0)

   # Step: 5: calculate mean step testing loss
   loss = total_error / len(data_loader.dataset)
   return loss, my_out, my_y


# Define the training process
def train(epochs, model, train_data_loader, val_data_loader, test_data_loader, optimizer, scheduler, device, disp_step, batch_size, learning_rate, numLayers, dispProgress):
   if dispProgress:
      # Prepare plotting
      plt.figure(figsize=(8, 6))
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.title('Loss vs. Epoch during Training')
      plt.grid(True)

   train_losses = []
   val_losses = []
   test_losses = []
   epoch_loss_list = []
   epochs_list = []

   prefix = 'Oxigraph'
   for epoch in range(1, epochs+1):
      model, train_loss = train_step(model, train_data_loader, optimizer, device)
      val_loss, val_out, val_y = test(model, val_data_loader, device)
      test_loss, test_out, test_y= test(model, test_data_loader, device)
      # Use val_loss to scheduler

      scheduler.step(val_loss) # This is for the ReduceLROnPlateau scheduler
      #scheduler.step()  # This is for the annealing scheduler
      epoch_loss_list.append(train_loss)

      # Display training progress
      if epoch % disp_step == 0:
         current_lr = optimizer.param_groups[0]['lr']
         progress_msg = 'Epoch '+ str(epoch) + \
                        ', training loss(MAE)=' + str(round(train_loss,4)) + \
                        ', Validating loss(MAE)=' + str(round(val_loss,4)) +  \
                        ', testing loss(MAE)=' + str(round(test_loss,4)) + \
                        ', lr=' + str(round(current_lr,8))
         print(progress_msg)
         train_losses.append(train_loss)
         val_losses.append(val_loss)
         test_losses.append(test_loss)

         epochs_list.append(epoch) 
         if dispProgress:
            plot_title = prefix 
            plot_training_progress(epoch, train_losses, val_losses, test_losses, title=plot_title)

   
   # Test the model after it's trained
   test_loss, test_out, test_y= test(model, test_data_loader, device)

   ymax = 5.10264
   ymin = 0.698339

   # reverse normalization
   test_out = test_out*(ymax - ymin) + ymin
   test_y = test_y*(ymax - ymin) + ymin


   get_print_agg_weights(model)

   # Save training results and information for later processing
   timestr = str(int(time.time()*1000000))
   folder = './Results/6_20'
   create_folder_if_not_exists(folder)


   maeFile = prefix +  '_MAE.txt'
   save_mae(folder=folder, maeFile=maeFile, test_loss=test_loss)
   progress_filename = prefix + '_' +  timestr + '_training_progress.txt'
   save_train_progress(folder, progress_filename, epochs=epochs_list, train_losses=train_losses, val_losses=val_losses, test_losses=test_losses)
   plotfilename = prefix + '_' + timestr + '_progress_plotting.png'
   plot_training_from_file(folder, filename=progress_filename, plotfilename=plotfilename, disp=False)
   agg_weight_filename = prefix + '_' + timestr + '_agg_weights.txt'
   save_agg_weights(folder, filename=agg_weight_filename, model=model)
   result_file = prefix + '_' + timestr + '_y.txt'
   save_result(folder, result_file, test_out = test_out, test_y = test_y)
   plot_regression(folder, result_file)

   model_file = prefix + '_' + timestr +'.pth'
   model_path = osp.join(folder, model_file)
   torch.save(model.state_dict(), model_path)




