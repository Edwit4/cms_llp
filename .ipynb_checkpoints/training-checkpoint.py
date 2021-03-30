import os
import pickle
import numpy as np
import torch
import random
from torch import nn
from sklearn.metrics import roc_curve, auc
from losses import llp_bce
from llp_auc import llp_roc_curve
from evaluation import plot_loss_histories, plot_pred_distribution, \
    plot_bag_histories, plot_ROC_curve, plot_AUC_history, \
    plot_llp_ROC_curve, plot_sig_frac, plot_param_sigmas

def train_model(model, data_loaders, num_epochs=100, learning_rate=1e-3,
        weight_decay=0, llp=False, reproducible=False, seed=123,
        early_stopping=False, min_change=0, patience=0,
        auc_num_thresh=1000, auc_const_denom=True, auc_output_thresh=True,
        image_prefix='image', output_prefix='output'):
    
    # Prep output filenames
    filenames = {'loss history':
                f'{image_prefix}_loss_history.png',

                'auc history': 
                f'{image_prefix}_auc_history.png',

                'valid ROC curve': 
                f'{image_prefix}_valid_ROC_curve.png',

                'valid output distribution': 
                f'{image_prefix}_valid_output_distribution.png',

                'bag history': 
                f'{image_prefix}_bag_history.png',
                
                'test ROC curve': 
                f'{image_prefix}_test_ROC_curve.png',
                
                'test output distribution':
                f'{image_prefix}_test_output_distribution.png',

                'test llp roc':
                f'{image_prefix}_test_llp_roc.png',

                'test sig frac': 
                f'{image_prefix}_test_sig_frac.png',

                'test frac sigma':
                f'{image_prefix}_test_frac_sigma.png',

                'model': 
                f'{output_prefix}_model.ckpt',

                'log': 
                f'{output_prefix}_log.pickle'} 

    # Use cuda if available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        
        # Make things as reproducible as possible. While results are close with
        # each run they still don't seem to be exact.
        if reproducible:
            # Benchmark can speed things up, but possibly also hurts
            # reproducibility
            torch.backends.cudnn.benchmark = False

            # Deterministic setting ensures that operations are done in the same
            # order every time, as floating point error accumulates differently
            # for different orders. Slows things down but increases
            # reproducibility.
            torch.backends.cudnn.deterministic = True

            # Manually set seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            np.random.seed(seed)  # Numpy module.
            random.seed(seed)  # Python random module.
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    model = model.to(device)

    # Use regular BCE for fully supervised case, or continuous bernoulli
    # version for weakly supervised case
    if llp:
        criterion = llp_bce
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
            weight_decay=weight_decay)

    total_step = len(data_loaders['train'])
    best_loss = None
    stop_early = False
    train_loss_history = []
    valid_loss_history = []
    auc_history = []
    fpr_history = []
    tpr_history = []
    out_history = []
    label_history = []
    if llp:
        valid_bag_loss = {}
        bag_history = {}
    for epoch in np.arange(num_epochs)+1:
        model.train()
        train_loss = 0.0
        # Train for weakly supervised case where proportion labels are
        # supplied
        if llp:
            for i, (images, labels, fractions) in \
                    enumerate(data_loaders['train']):
                # Forward pass
                images, labels, fractions = images.to(device), \
                    labels.to(device), fractions.to(device)
                outputs = model(images).to(device)
                loss = criterion(outputs, fractions)
                train_loss += loss.item()

                # Backward pass and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print metrics
                if (i + 1) % 50 == 0:
                    print(f'Epoch {epoch}/{num_epochs}, Step {i+1}/{total_step}, Loss: {loss.item():.3f}\r',end='')
                elif i == len(data_loaders['train']) - 1:
                    print(f'Epoch {epoch}/{num_epochs}, Step {i+1}/{total_step}, Loss: {loss.item():.3f}\r',end='')

        # Train for fully supervised case
        else:
            for i, (images, labels) in \
                    enumerate(data_loaders['train']):
                # Forward pass
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).to(device)
                outputs = outputs.reshape(outputs.shape[0])
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                # Backward pass and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print metrics
                if (i + 1) % 50 == 0:
                    print(f'Epoch {epoch}/{num_epochs}, Step {i+1}/{total_step}, Loss: {loss.item():.3f}\r',end='')
                elif i == len(data_loaders['train']) - 1:
                    print(f'Epoch {epoch}/{num_epochs}, Step {i+1}/{total_step}, Loss: {loss.item():.3f}\r',end='')

        train_loss /= len(data_loaders['train'])
        print('')

        # Track training loss history
        train_loss_history.append(train_loss)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            all_outputs = []
            all_labels = []
            val_loss = 0.0
            if llp:
                for images, labels, fractions in data_loaders['valid']:
                    images, labels, fractions = images.to(device), \
                            labels.to(device), fractions.to(device)
                    outputs = model(images).to(device)
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                    loss = criterion(outputs, fractions)
                    val_loss += loss.item()

                    f = fractions[0].item()
                    try:
                        valid_bag_loss[str(f)]['value'] += loss.item()
                        valid_bag_loss[str(f)]['count'] += 1
                    except:
                        valid_bag_loss[str(f)] = {'value': loss.item(),
                                'count':1}

                for key in valid_bag_loss.keys():
                    valid_bag_loss[key]['value'] /= valid_bag_loss[key]['count']
                    try:
                        bag_history[key].append(valid_bag_loss[key]['value'])
                    except:
                        bag_history[key] = [valid_bag_loss[key]['value']]

                # Track validation loss history
                val_loss /= len(data_loaders['valid'])
                valid_loss_history.append(val_loss)

                for key in valid_bag_loss.keys():
                    valid_bag_loss[key]['count'] = 0

                plot_bag_histories(bag_history, filenames['bag history'])

            else:
                for images, labels in data_loaders['valid']:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).to(device)
                    outputs = outputs.reshape(outputs.shape[0])
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                # Track validation loss history
                val_loss /= len(data_loaders['valid'])
                valid_loss_history.append(val_loss)

        if early_stopping:
            if best_loss == None:
                best_loss = val_loss
                patience_counter = 0
                state = {'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss}
                checkpoint = torch.save(state, 'checkpoint.ckpt')
            elif val_loss > best_loss - min_change:
                patience_counter += 1
                if patience_counter >= patience:
                    stop_early = True
            else:
                best_loss = val_loss
                patience_counter = 0
                state = {'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss}
                checkpoint = torch.save(state, 'checkpoint.ckpt')

        all_outputs = np.concatenate(
                [output.cpu().numpy() for output in all_outputs])
        all_labels = np.concatenate(
                [label.cpu().numpy() for label in all_labels])
        plot_pred_distribution(all_labels, all_outputs, 
                filenames['valid output distribution'])

        fpr, tpr, thresh = roc_curve(all_labels, all_outputs)
        area = auc(fpr, tpr)
        auc_history.append(area)
        fpr_history.append(fpr)
        tpr_history.append(tpr)
        out_history.append(all_outputs)
        label_history.append(all_labels)

        print(f'train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, val_auc: {area:.3f}\n')

        plot_AUC_history(auc_history, filenames['auc history'])

        plot_loss_histories(train_loss_history, valid_loss_history, 
                filenames['loss history'])

        plot_ROC_curve(all_labels, all_outputs, 
                filenames['valid ROC curve'])

        log = {'train loss': train_loss_history,
                'valid loss': valid_loss_history,
                'auc': auc_history,
                'fpr': fpr_history,
                'tpr': tpr_history,
                'out': out_history,
                'label': label_history}

        if stop_early:
            stop_epoch = epoch-patience
            print(f'Stopped early, saving model from epoch {stop_epoch}')
            plot_loss_histories(train_loss_history, valid_loss_history, 
                    filenames['loss history'],early_stopping=stop_epoch)
            checkpoint = torch.load('checkpoint.ckpt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            break

    # Check if test set is present
    try:
        data_loaders['test']
        test_set_exists = True
    except KeyError:
        test_set_exists = False
        
    # If there is a test set, see how model performs on it
    if test_set_exists:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            all_outputs = []
            all_labels = []
            all_Ms = []
            if llp:
                for images, labels, fractions, Ms in data_loaders['test']:
                    images, labels, fractions, Ms = images.to(device), \
                            labels.to(device), fractions.to(device), \
                            Ms.to(device)
                    outputs = model(images).to(device)
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                    all_Ms.append(Ms)
            else:
                for images, labels in data_loaders['test']:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).to(device)
                    all_outputs.append(outputs.data)
                    all_labels.append(labels)

            all_outputs = np.concatenate(
                    [output.cpu().numpy() for output in all_outputs])
            all_labels = np.concatenate(
                    [label.cpu().numpy() for label in all_labels])
            output_shape = all_outputs.shape
            all_labels = all_labels.reshape(output_shape)
            if llp:
                all_Ms = np.concatenate(
                        [M.cpu().numpy() for M in all_Ms])
                all_Ms = all_Ms.reshape(output_shape)

            fpr, tpr, thresh, area = plot_ROC_curve(
                    all_labels, all_outputs, filenames['test ROC curve'])
            opt_index = np.argmax(tpr - fpr + 1)
            opt_thresh = thresh[opt_index]
            bool_pred = np.int64(all_outputs > opt_thresh)
            correct = np.int64(bool_pred == all_labels)
            accuracy = np.sum(correct) / len(correct) * 100

            print(f'test auc: {area}')
            print(f'optimal threshold: {opt_thresh}')
            print(f'accuracy for optimal threshold: {accuracy} percent')

            plot_pred_distribution(all_labels, all_outputs,
                    filenames['test output distribution'])

            if llp:
                roc_dict = llp_roc_curve(all_outputs, all_Ms,
                        seed=seed, num_thresh=auc_num_thresh, 
                        constant_denom=auc_const_denom, output_thresh=auc_output_thresh)

                plot_llp_ROC_curve(roc_dict['fpr'], roc_dict['tpr'],
                        filenames['test llp roc'])
                plot_sig_frac(roc_dict['threshold'], 
                        roc_dict['upper fit dict']['params']['frac'],
                        filenames['test sig frac'])
                plot_param_sigmas(roc_dict['threshold'], 
                        roc_dict['upper fit dict']['sigmas']['frac'], 
                    filenames['test frac sigma'])

                log['test'] = {'auc': area, 
                        'fpr': fpr,
                        'tpr': tpr,
                        'out': all_outputs,
                        'label': all_labels,
                        'llp roc': roc_dict}

            else:
                log['test'] = {'auc': area, 
                        'fpr': fpr,
                        'tpr': tpr,
                        'out': all_outputs,
                        'label': all_labels}

    # Save the model and log file
    torch.save(model.state_dict(), f'{filenames["model"]}')
    with open(f'{filenames["log"]}','wb') as handle:
        pickle.dump(log, handle)

    return model, log 
