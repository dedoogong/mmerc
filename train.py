from mmerc.textFeature.textFeatures import DecoderRNN as decoder, EncoderCNN as encoder
import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
opt='SGD'

if opt == 'SGD':
    optimizer=optim.SGD(
        #self.model.parameters(),
        #lr=self.arg.base_lr,
        #momentum=0.9,
        #nesterov=self.arg.nesterov,
        #weight_decay=self.arg.weight_decay)
    )
elif opt == 'Adam':
    optimizer=optim.Adam(
        #self.model.parameters(),
        #lr=self.arg.base_lr,
        #weight_decay=self.arg.weight_decay)
    )

criterion=1000
train_data_loader=1000
val_data_loader=1000
device=torch.device('cuda')
num_epochs=5
total_step=1000000
vocab_size=1000
save_every = 1000
# get the losses for vizualization
losses = list()
val_losses = list()
loss = nn.CrossEntropyLoss()
for epoch in range(1, 10+1):

    for i_step in range(1, total_step+1):

        # zero the gradients
        decoder.zero_grad()
        encoder.zero_grad()

        # set decoder and encoder into train mode
        encoder.train()
        decoder.train()

        # Randomly sample a caption length, and sample indices with that length.
        indices = train_data_loader.dataset.get_train_indices()

        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_data_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch.
        images, captions = next(iter(train_data_loader))

        # make the captions for targets and teacher forcer
        captions_target = captions[:, 1:].to(device)
        captions_train = captions[:, :captions.shape[1]-1].to(device)

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)

        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions_train)

        # Calculate the batch loss
        loss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))

        # Backward pass
        loss.backward()

        # Update the parameters in the optimizer
        optimizer.step()

        # - - - Validate - - -
        # turn the evaluation mode on
        with torch.no_grad():

            # set the evaluation mode
            encoder.eval()
            decoder.eval()

            # get the validation images and captions
            val_images, val_captions = next(iter(val_data_loader))

            # define the captions
            captions_target = val_captions[:, 1:].to(device)
            captions_train = val_captions[:, :val_captions.shape[1]-1].to(device)

            # Move batch of images and captions to GPU if CUDA is available.
            val_images = val_images.to(device)

            # Pass the inputs through the CNN-RNN model.
            features = encoder(val_images)
            outputs = decoder(features, captions_train)

            # Calculate the batch loss.
            val_loss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))

        # append the validation loss and training loss
        val_losses.append(val_loss.item())
        losses.append(loss.item())

        # save the losses
        np.save('losses', np.array(losses))
        np.save('val_losses', np.array(val_losses))

        # Get training statistics.
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Val Loss: %.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), val_loss.item())

        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()

    # Save the weights.
    if epoch % save_every == 0:
        print("\nSaving the model")
        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pth' % epoch))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pth' % epoch))