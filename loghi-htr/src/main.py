# Imports

# > Standard library
from __future__ import division, print_function
import json
import logging
import random
import re
import subprocess
import uuid

# > Local dependencies
from arg_parser import get_args
from data_loader import DataLoader
from model import replace_final_layer, replace_recurrent_layer, train_batch
from utils import Utils, normalize_confidence, decode_batch_predictions, \
    load_model_from_directory
from vgsl_model_generator import VGSLModelGenerator
from custom_layers import ResidualBlock

# > Third party dependencies
import editdistance
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from word_beam_search import WordBeamSearch

# > Environment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def set_deterministic(args):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def remove_tags(text):
    # public static String STRIKETHROUGHCHAR = "␃"; //Unicode Character “␃” (U+2403)
    text = text.replace('␃', '')
    # public static String UNDERLINECHAR = "␅"; //Unicode Character “␅” (U+2405)
    text = text.replace('␅', '')
    # public static String SUBSCRIPTCHAR = "␄"; // Unicode Character “␄” (U+2404)
    text = text.replace('␄', '')
    # public static String SUPERSCRIPTCHAR = "␆"; // Unicode Character “␆” (U+2406)
    text = text.replace('␆', '')
    return text


def main():
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = get_args()
    if args.deterministic:
        set_deterministic(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # place from/imports here so os.environ["CUDA_VISIBLE_DEVICES"]  is set before TF loads
    from model import CERMetric, WERMetric, CTCLoss
    import tensorflow.keras as keras
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    if not args.use_float32 and args.gpu != '-1':
        print("using mixed_float16")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print("using float32")

    learning_rate = args.learning_rate

    if args.output and not os.path.exists(args.output):
        try:
            os.makedirs(args.output)
        except OSError as error:
            print(error)
            print('can not create output directory')

    char_list = None
    charlist_location = args.charlist
    if not charlist_location and args.existing_model:
        charlist_location = args.existing_model + '/charlist.txt'
    elif not charlist_location:
        charlist_location = args.output + '/charlist.txt'
    model_channels = args.channels
    model_height = args.height
    with strategy.scope():
        if args.existing_model:
            if not os.path.exists(args.existing_model):
                print('cannot find existing model on disk: ' + args.existing_model)
                exit(1)
            if not os.path.exists(charlist_location) and not args.replace_final_layer:
                print('cannot find charlist on disk: ' + charlist_location)
                exit(1)
            if not args.replace_final_layer:
                with open(charlist_location) as file:
                    char_list = list(char for char in file.read())
                print("using charlist")
                print("length charlist: " + str(len(char_list)))
                print(char_list)

            custom_objects = {
                'CERMetric': CERMetric,
                'WERMetric': WERMetric,
                'CTCLoss': CTCLoss,
                'ResidualBlock': ResidualBlock
            }
            model = load_model_from_directory(args.existing_model,
                                              custom_objects=custom_objects)

            if args.model_name:
                model._name = args.model_name

            if args.use_float32 or args.gpu == '-1':
                # Recreate the exact same model but with float32
                config = model.get_config()

                # Set the dtype policy for each layer in the configuration
                for layer_config in config['layers']:
                    if 'dtype' in layer_config['config']:
                        layer_config['config']['dtype'] = 'float32'
                    if 'dtype_policy' in layer_config['config']:
                        layer_config['config']['dtype_policy'] = {
                            'class_name': 'Policy',
                            'config': {'name': 'float32'}}

                # Create a new model from the modified configuration
                model_new = keras.Model.from_config(config)
                model_new.set_weights(model.get_weights())

                model = model_new

                # Verify float32
                for layer in model.layers:
                    assert layer.dtype_policy.name == 'float32'

            if not args.replace_final_layer:
                model_channels = model.layers[0].input_shape[0][3]

            model_height = model.layers[0].input_shape[0][2]
            if args.height != model_height:
                print(
                    'input height differs from model channels. use --height ' + str(model_height))
                print('resetting height to: ' + str(model_height))
                args.__dict__['height'] = model_height
                if args.no_auto:
                    exit(1)
            if not args.replace_final_layer:
                with open(charlist_location) as file:
                    char_list = list(char for char in file.read())
        img_size = (model_height, args.width, model_channels)
        loader = DataLoader(args.batch_size, img_size,
                            train_list=args.train_list,
                            validation_list=args.validation_list,
                            test_list=args.test_list,
                            inference_list=args.inference_list,
                            char_list=char_list,
                            do_binarize_sauvola=args.do_binarize_sauvola,
                            do_binarize_otsu=args.do_binarize_otsu,
                            multiply=args.multiply,
                            augment=args.augment,
                            elastic_transform=args.elastic_transform,
                            random_crop=args.random_crop,
                            random_width=args.random_width,
                            check_missing_files=args.check_missing_files,
                            distort_jpeg=args.distort_jpeg,
                            replace_final_layer=args.replace_final_layer,
                            normalization_file=args.normalization_file,
                            use_mask=args.use_mask,
                            do_random_shear=args.do_random_shear
                            )

        print("creating generators")
        training_generator, validation_generator, test_generator, inference_generator, utilsObject, train_batches = loader.generators()

        # Testing
        print(len(loader.charList))

        if args.decay_rate > 0 and args.decay_steps > 0:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=args.decay_steps,
                decay_rate=args.decay_rate)
        elif args.decay_rate > 0 and args.decay_steps == -1 and args.do_train:
            if training_generator is None:
                print(
                    'training, but training_generator is None. Did you provide a train_list?')
                exit(1)
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=train_batches,
                decay_rate=args.decay_rate)
        else:
            lr_schedule = learning_rate

        output_charlist_location = args.output_charlist
        if not output_charlist_location:
            output_charlist_location = args.output + '/charlist.txt'

        if args.existing_model:
            print('using existing model as base: ' + args.existing_model)

            if args.replace_recurrent_layer:
                model = replace_recurrent_layer(model,
                                                len(char_list),
                                                args.replace_recurrent_layer,
                                                use_mask=args.use_mask)

            if args.replace_final_layer:
                with open(output_charlist_location, 'w') as chars_file:
                    chars_file.write(str().join(loader.charList))
                char_list = loader.charList

                model = replace_final_layer(model, len(
                    char_list), model.name, use_mask=args.use_mask)

            if any([args.thaw, args.freeze_conv_layers,
                    args.freeze_recurrent_layers, args.freeze_dense_layers]):
                for layer in model.layers:
                    if args.thaw:
                        layer.trainable = True
                    elif args.freeze_conv_layers and (layer.name.lower().startswith("conv") or layer.name.lower().startswith("residual")):
                        print(layer.name)
                        layer.trainable = False
                    elif args.freeze_recurrent_layers and layer.name.lower().startswith("bidirectional"):
                        print(layer.name)
                        layer.trainable = False
                    elif args.freeze_dense_layers and layer.name.lower().startswith("dense"):
                        print(layer.name)
                        layer.trainable = False

        else:
            # save characters of model for inference mode
            with open(output_charlist_location, 'w') as chars_file:
                chars_file.write(str().join(loader.charList))

            char_list = loader.charList

            print("creating new model")
            model_generator = VGSLModelGenerator(
                model=args.model,
                name=args.model_name,
                channels=model_channels,
                output_classes=len(char_list) + 2
                if args.use_mask else len(char_list) + 1
            )

            model = model_generator.build()

        optimizers = {
            "adam": keras.optimizers.Adam,
            "adamw": keras.optimizers.experimental.AdamW,
            "adadelta": keras.optimizers.Adadelta,
            "adagrad": keras.optimizers.Adagrad,
            "adamax": keras.optimizers.Adamax,
            "adafactor": keras.optimizers.Adafactor,
            "nadam": keras.optimizers.Nadam
        }

        if args.optimizer in optimizers:
            model.compile(optimizers[args.optimizer](learning_rate=lr_schedule),
                          loss=CTCLoss,
                          metrics=[CERMetric(greedy=args.greedy,
                                             beam_width=args.beam_width),
                                   WERMetric()])
        else:
            print('wrong optimizer')
            exit()

    model.summary(line_length=110)

    model_channels = model.layers[0].input_shape[0][3]
    if args.channels != model_channels:
        print('input channels differs from model channels. use --channels ' +
              str(model_channels))
        if args.no_auto:
            exit()
        else:
            args.__dict__['channels'] = model_channels
            print('setting correct number of channels: ' + str(model_channels))

    model_outputs = model.layers[-1].output_shape[2]
    num_characters = len(char_list) + 1
    if args.use_mask:
        num_characters = num_characters + 1
    if model_outputs != num_characters:
        print('model_outputs: ' + str(model_outputs))
        print('charlist: ' + str(num_characters))
        print('number of characters in model is different from charlist provided.')
        print('please find correct charlist and use --charlist CORRECT_CHARLIST')
        print('if the charlist is just 1 lower: did you forget --use_mask')
        exit(1)

    if args.do_train:
        validation_dataset = None
        if args.validation_list:
            validation_dataset = validation_generator

        store_info(args, model)

        training_dataset = training_generator
        print('batches ' + str(training_dataset.__len__()))
        # try:
        metadata = get_config(args, model)
        history = train_batch(
            model,
            training_dataset,
            validation_dataset,
            epochs=args.epochs,
            output=args.output,
            model_name=model.name,
            steps_per_epoch=args.steps_per_epoch,
            max_queue_size=args.max_queue_size,
            early_stopping_patience=args.early_stopping_patience,
            output_checkpoints=args.output_checkpoints,
            charlist=loader.charList,
            metadata=metadata,
            verbosity_mode=args.training_verbosity_mode
        )

        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history["loss"], label="loss")
        if args.validation_list:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/CER")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(args.output, 'plot.png'))

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history["CER_metric"], label="CER train")
        if args.validation_list:
            plt.plot(history.history["val_CER_metric"], label="CER val")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/CER")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(args.output, 'plot2.png'))

        # except tf.python.framework.errors_impl.ResourceExhaustedError as e:
        #     print("test")
    if args.do_validate:
        print("do_validate")
        utilsObject = Utils(char_list, args.use_mask)
        validation_dataset = validation_generator

        # Get the prediction model by taking the last dense layer of the full
        # model
        last_dense_layer = None
        for layer in reversed(model.layers):
            if layer.name.startswith('dense'):
                last_dense_layer = layer
                break

        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, last_dense_layer.output
        )
        prediction_model.summary(line_length=110)

        wbs = None
        if args.corpus_file:
            if not os.path.exists(args.corpus_file):
                print('cannot find corpus_file on disk: ' + args.corpus_file)
                exit(1)

            with open(args.corpus_file) as f:
                corpus = ''
                for line in f:
                    if args.normalization_file:
                        line = loader.normalize(line)
                    corpus += line
            word_chars = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzßàáâçèéëïñôöûüň'
            chars = '' + ''.join(sorted(list(char_list)))

            print('using corpus file: ' + str(args.corpus_file))
            wbs = WordBeamSearch(args.beam_width, 'NGrams', args.wbs_smoothing, corpus.encode('utf8'), chars.encode('utf8'),
                                 word_chars.encode('utf8'))
            print('Created WordBeamSearcher')

        total_cer = total_cer_lower = total_cer_simple = 0
        total_length = total_length_simple = 0

        norm_total_cer = norm_total_cer_lower = 0
        norm_total_length = 0

        total_cer_wbs = total_cer_wbs_lower = total_cer_wbs_simple = 0
        batch_no = pred_counter = 0

        for batch in validation_dataset:
            predictions = prediction_model.predict(batch[0])
            predicted_texts = decode_batch_predictions(predictions, utilsObject, args.greedy, args.beam_width,
                                                       args.num_oov_indices)
            predsbeam = tf.transpose(predictions, perm=[1, 0, 2])

            if wbs:
                print('computing wbs...')
                label_str = wbs.compute(predsbeam)
                char_str = [''.join([chars[label] for label in curr_label_str]) for curr_label_str in label_str]
                [print(s) for s in char_str]

            orig_texts = [tf.strings.reduce_join(utilsObject.num_to_char(label)).numpy().decode("utf-8").strip() for
                          label in batch[1]]

            for prediction in predicted_texts:
                for i in range(len(prediction)):
                    confidence, predicted_text, original_text = prediction[i][0], prediction[i][1], orig_texts[i]

                    original_text = original_text.strip().replace('', '')
                    predicted_text = predicted_text.strip().replace('', '')
                    original_text = remove_tags(original_text)
                    predicted_text = remove_tags(predicted_text)

                    ground_truth_simple = re.sub('[\W_]+', '', original_text).lower()
                    predicted_simple = re.sub('[\W_]+', '', predicted_text).lower()

                    cer = editdistance.eval(original_text, predicted_text) / max(len(original_text), 1)
                    current_editdistance = editdistance.eval(original_text, predicted_text)
                    current_editdistance_lower = editdistance.eval(original_text.lower(), predicted_text.lower())
                    current_editdistance_simple = editdistance.eval(ground_truth_simple, predicted_simple)

                    if wbs:
                        predicted_wbs_simple = re.sub('[\W_]+', '', char_str[i]).lower()
                        current_editdistance_wbs_simple = editdistance.eval(ground_truth_simple, predicted_wbs_simple)
                        current_editdistance_wbs = editdistance.eval(original_text, char_str[i].strip())
                        current_editdistance_wbs_lower = editdistance.eval(original_text.lower(),
                                                                           char_str[i].strip().lower())

                        total_cer_wbs_simple += current_editdistance_wbs_simple
                        total_cer_wbs += current_editdistance_wbs
                        total_cer_wbs_lower += current_editdistance_wbs_lower

                    #Apply normalisation of ground truth + prediction and calculate normalised CER
                    if args.normalization_file:
                        #Apply normalisation from DataLoader
                        original_text_normalised = loader.normalize(original_text, args.normalization_file)
                        predicted_text_normalised = loader.normalize(predicted_text, args.normalization_file)

                        #Calculate editdistance between original_text and pred
                        norm_current_editdistance = editdistance.eval(original_text_normalised,
                                                                      predicted_text_normalised)

                        # Calculate editdistance on lowercase original_text and pred
                        norm_current_editdistance_lower = editdistance.eval(original_text_normalised.lower(),
                                                                            predicted_text_normalised.lower())

                        #Aggregate cer
                        norm_total_cer += norm_current_editdistance
                        norm_total_cer_lower += norm_current_editdistance_lower
                        norm_total_length += len(original_text_normalised)

                    if cer > 0.0:
                        filename = loader.get_item('validation', (batch_no * args.batch_size) + i)
                        [print(f'\n{filename}\n{original_text}\n{predicted_text}') for _ in range(3)]
                        if wbs:
                            [print(s) for s in char_str]

                    total_cer += current_editdistance
                    total_cer_lower += current_editdistance_lower
                    total_cer_simple += current_editdistance_simple
                    total_length += len(original_text)
                    total_length_simple += len(ground_truth_simple)

                    if cer > 0.0:
                        print(f'confidence: {confidence} cer: {cer} total_orig: {len(original_text)} total_pred: {len(predicted_text)} errors: {current_editdistance}')
                        print(f'avg editdistance: {total_cer / total_length}')
                        print(f'avg editdistance lower: {total_cer_lower / total_length}')

                        if total_length_simple > 0:
                            print(f'avg editdistance simple: {total_cer_simple / total_length_simple}')

                        if args.normalization_file:
                            print(f'avg norm_editdistance: {norm_total_cer / norm_total_length}')
                            print(f'avg norm_editdistance lower: {norm_total_cer_lower / norm_total_length}')

                        if wbs:
                            print(f'avg editdistance wbs: {total_cer_wbs / total_length}')
                            print(f'avg editdistance wbs lower: {total_cer_wbs_lower / total_length}')

                            if total_length_simple > 0:
                                print(f'avg editdistance wbs_simple: {total_cer_wbs_simple / total_length_simple}')
                    else:
                        print('.', end='')
                    pred_counter += 1
            batch_no += 1

        #Calculate averages
        total_cer /= total_length
        total_cer_lower /= total_length
        total_cer_simple /= total_length_simple

        if args.normalization_file:
            norm_total_cer /= norm_total_length
            norm_total_cer_lower /= norm_total_length

            total_norm_cer_lower = round(norm_total_cer - calc_95_confidence_interval(norm_total_cer, pred_counter), 4)
            total_norm_cer_upper = round(norm_total_cer + calc_95_confidence_interval(norm_total_cer, pred_counter), 4)

            total_norm_cer_lower_lower = round(
                norm_total_cer_lower - calc_95_confidence_interval(norm_total_cer_lower, pred_counter), 4)
            total_norm_cer_lower_upper = round(
                norm_total_cer_lower + calc_95_confidence_interval(norm_total_cer_lower, pred_counter), 4)



        if wbs:
            total_cer_wbs_simple /= total_length_simple
            total_cer_wbs /= total_length
            total_cer_wbs_lower /= total_length

        #Calculate confidence ranges upper/lower bounds
        total_cer_lowerbound = round(total_cer - calc_95_confidence_interval(total_cer, pred_counter), 4)
        total_cer_upperbound = round(total_cer + calc_95_confidence_interval(total_cer, pred_counter), 4)
        total_cer_lower_lowerbound = round(total_cer_lower - calc_95_confidence_interval(total_cer_lower, pred_counter), 4)
        total_cer_lower_upperbound = round(total_cer_lower + calc_95_confidence_interval(total_cer_lower, pred_counter), 4)
        total_cer_simple_lowerbound = round(total_cer_simple - calc_95_confidence_interval(total_cer_simple, pred_counter),4)
        total_cer_simple_upperbound = round(total_cer_simple + calc_95_confidence_interval(total_cer_simple, pred_counter),4)

        print("--------------------------------------------------------")
        print(f'totalcer: {total_cer} (95% conf. range [{total_cer_lowerbound},{total_cer_upperbound}])')
        print(f'totalcerlower: {total_cer_lower} (95% conf. range [{total_cer_lower_lowerbound},{total_cer_lower_upperbound}])')
        print(f'totalcersimple: {total_cer_simple} (95% conf. range [{total_cer_simple_lowerbound},{total_cer_simple_upperbound}])')
        print("--------------------------------------------------------")

        if args.normalization_file:
            print(f'norm_totalcer: {norm_total_cer} (95% conf. range [{total_norm_cer_lower},{total_norm_cer_upper}])')
            print(f'norm_totalcer_lower: {norm_total_cer_lower} (95% conf. range [{total_norm_cer_lower_lower},{total_norm_cer_lower_upper}])')
            print("--------------------------------------------------------")

        if wbs:
            print(f'totalcerwbs: {total_cer_wbs}')
            print(f'totalcerwbslower: {total_cer_wbs_lower}')
            print(f'totalcerwbssimple: {total_cer_wbs_simple}')

    if args.do_inference:
        print('inferencing')
        print(char_list)
        loader = DataLoader(args.batch_size,
                            img_size,
                            char_list,
                            inference_list=args.inference_list,
                            check_missing_files=args.check_missing_files,
                            normalization_file=args.normalization_file,
                            use_mask=args.use_mask
                            )
        training_generator, validation_generator, test_generator, inference_generator, utilsObject, train_batches = loader.generators()

        # Get the prediction model by taking the last dense layer of the full
        # model
        last_dense_layer = None
        for layer in reversed(model.layers):
            if layer.name.startswith('dense'):
                last_dense_layer = layer
                break

        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, last_dense_layer.output
        )

        prediction_model.summary(line_length=110)

        inference_dataset = inference_generator

        store_info(args, model)

        batch_counter = 0
        with open(args.results_file, "w") as text_file:

            for batch in inference_dataset:
                predictions = prediction_model.predict_on_batch(batch[0])
                predicted_texts = decode_batch_predictions(
                    predictions, utilsObject, args.greedy, args.beam_width)
                orig_texts = []
                for label in batch[1]:
                    label = tf.strings.reduce_join(
                        utilsObject.num_to_char(label)).numpy().decode("utf-8")
                    orig_texts.append(label.strip())
                for prediction in predicted_texts:
                    for i in range(len(prediction)):
                        confidence = prediction[i][0]
                        predicted_text = prediction[i][1]
                        filename = loader.get_item(
                            'inference', (batch_counter * args.batch_size) + i)
                        original_text = orig_texts[i].strip().replace('', '')
                        predicted_text = predicted_text.strip().replace('', '')

                        confidence = normalize_confidence(
                            confidence, predicted_text)

                        print(original_text)
                        print(filename + "\t" + str(confidence) +
                              "\t" + predicted_text)
                        text_file.write(
                            filename + "\t" + str(confidence) + "\t" + predicted_text + "\n")

                    batch_counter += 1
                    text_file.flush()


def get_config(args, model):
    if os.path.exists("version_info"):
        with open("version_info") as file:
            version_info = file.read()
    else:
        bash_command = 'git log --format="%H" -n 1'
        process = subprocess.Popen(
            bash_command.split(), stdout=subprocess.PIPE)
        output, errors = process.communicate()
        version_info = output.decode(
            'utf8', errors='strict').strip().replace('"', '')

    model_layers = []
    model.summary(print_fn=lambda x: model_layers.append(x))

    config = {
        'git_hash': version_info.strip(),
        'args': args.__dict__,
        'model': model_layers,
        'notes': ' ',
        'uuid': str(uuid.uuid4())
    }
    return config


def store_info(args, model):
    config = get_config(args, model)
    if args.config_file_output:
        config_file_output = args.config_file_output
    else:
        config_file_output = os.path.join(args.output, 'config.json')
    with open(config_file_output, 'w') as configuration_file:
        json.dump(config, configuration_file)


def calc_95_confidence_interval(cer_metric, n):
    """ Calculates the binomial confidence radius of the given metric
    based on the num of samples (n) and a 95% certainty number
    E.g. cer_metric = 0.10, certainty = 95 and n= 5500 samples -->
    conf_radius = 1.96 * ((0.1*(1-0.1))/5500)) ** 0.5 = 0.008315576
    This means with 95% certainty we can say that the True CER of the model is between 0.0917 and 0.1083 (4-dec rounded)
    """
    return 1.96 * ((cer_metric*(1-cer_metric))/n) ** 0.5


if __name__ == '__main__':
    main()
