from .data_preprocessor import DataPreprocessor
from .car_cutter import CarCutter
from .train import BetaVAETrainer
from nuimages import NuImages
from .gan import GAN
from .model_eval import ModelEval
from matplotlib import pyplot as plt
import torch
import json

class AutoDoppelGANgerShell:
    def __init__(self):
        print("Welcome to AutoDoppelGANger. Type 'help' to list commands.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GAN(64, 64, 3, 100, device, "logs")
        self.model_vae = BetaVAETrainer(3, 128, [32, 64, 128, 256, 512], device, "logs", "models/checkpoints/")
        self.model_eval = ModelEval(32, device)

    def help(self):
        print("Available commands:")
        print("lddst - Load dataset (Usage: lddst <filepath>)")
        print("gensam - Generate Samples (Usage: gensam <type> <num_samples> <num_rows> <num_columns>)")
        print("ldwghtsgan - Load model weights (Usage: ldwghtsgan <path to discriminator weights> <path to generator weights>)")
        print("ldwghtsvae - Load model weights (Usage: ldwghtsvae <weights filename>)")
        print("traingan - Start training the GAN model (Usage: traingan <path to json file>)")
        print("trainbvae - Start training the Beta-VAE model (Usage: trainbvae <path to json file>)")
        print("incsc - Calculate inception score of the model")
        # print("fid - Calculate FID of the model")
        print("exit - Exit the shell")

    def cut_nu_images(self, dataroot, out_path, version, min_size_x, min_size_y):
        print("Cutting out cars...")
        nuim = NuImages(dataroot=dataroot, version=version, verbose=False, lazy=True)
        cutter = CarCutter(nuim, min_size_x, min_size_y)
        cutter.cut_out_vehicles_from_dataset(dataroot, out_path+"car_cut/")
        print("Done.")

    def load_dataset(self, filepath, target_width=64, target_height=64, img_channels=3):
        print("Loading dataset...")
        try:
            preprocess = DataPreprocessor(target_width, target_height, img_channels)
            self.dataset = preprocess.load_dataset(filepath)
            print("Done.")
        except AttributeError as e:
            print(f"Attribute error: {e}")
        except ImportError as e:
            print(f"Import error: {e}")
        except TypeError as e:
            print(f"Type error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except RuntimeError as e:
            print(f"Runtime error: {e}")
        except OSError as e:
            print(f"OS error: {e}")

    def train_gan(self, json_file):
        print("Training model...")
        try:
            with open(json_file, 'r') as file_handle:
                training_setup = json.load(file_handle)

            if not self.dataset:
                print("You must load dataset with 'lddst <filepath>' before training model.")
                return
            self.model.train(self.dataset, training_setup["num_epochs"], training_setup["batch_size"], training_setup["learning_rate"],
                            training_setup["beta1"], training_setup["beta2"], training_setup["time_limit"])
            if training_setup["save_weights"]:
                discriminator = "models/checkpoints/" + training_setup["discriminator_weights_filename"] +".pth"
                generator = "models/checkpoints/" + training_setup["generator_weights_filename"] +".pth"
                self.model.save_models_weigths(discriminator, generator)
        except FileNotFoundError as e:
            print(f"Configuration file not found: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from configuration file: {e}")
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
        except AttributeError as e:
            print(f"Attribute error: {e}")
        except TypeError as e:
            print(f"Type error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        print("Done.")

    def train_bvae(self, json_file):
        print("Training model...")
        try:
            with open(json_file, 'r') as file_handle:
                training_setup = json.load(file_handle)

            if not self.dataset:
                print("You must load dataset with 'lddst <filepath>' before training model.")
                return
            self.model_vae.train(self.dataset, training_setup["num_epochs"], training_setup["batch_size"], training_setup["beta"],
                            training_setup["learning_rate"], training_setup["weights_filename"])
        except FileNotFoundError as e:
            print(f"Configuration file not found: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from configuration file: {e}")
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
        except AttributeError as e:
            print(f"Attribute error: {e}")
        except TypeError as e:
            print(f"Type error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        print("Done.")

    def generate_samples(self, type, num_samples, num_rows, num_cols):
        print("Displaying images...")
        try:
            num_samples = int(num_samples)
            num_rows = int(num_rows)
            num_cols = int(num_cols)
            if type =="gan":
                images = self.model.generate_samples(num_samples)
            else:
                images = self.model_vae.generate_samples(num_samples)
            _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i].permute(1, 2, 0).clamp(0, 1))
                ax.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()
            print("Done...")
        except ValueError:
            print("Pass correct arguments: <type(gan/bvae)> <num_samples> <num_rows> <num_columns>")

    def generate_disentangled_samples(self, num_samples, num_rows, num_columns):
        print("Displaying images...")
        try:
            num_samples = int(num_samples)
            num_rows = int(num_rows)
            num_cols = int(num_columns)
            base_vector = torch.randn(num_samples, 128)
            images = self.model_vae.generate_disentangled_samples(base_vector)
            _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
            for i, ax_row in enumerate(axes):
                for j, ax in enumerate(ax_row):
                    ax.imshow(images[i*10+j])
                    ax.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()
        except ValueError:
            print("Pass correct arguments: <num_samples> <num_rows> <num_columns>")
        except TypeError:
            print("Rows and columns need to be bigger than one")

    def load_weights_gan(self, path_dsc, path_gen):
        try:
            self.model.load_model_weights(path_dsc, path_gen)
            print("Done")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except (torch.serialization.pickle.UnpicklingError, EOFError) as e:
            print(f"Unpickling error: {e}")
        except RuntimeError as e:
            print(f"Runtime error: {e}")
        except IsADirectoryError:
            print(f"Weight must be file: {e}")

    def load_weights_vae(self, name):
        try:
            self.model_vae.load_model_weights(name)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except (torch.serialization.pickle.UnpicklingError, EOFError) as e:
            print(f"Unpickling error: {e}")
        except RuntimeError as e:
            print(f"Runtime error: {e}")
        except IsADirectoryError:
            print(f"Weight must be file: {e}")

    def display_inception_score(self):
        print("Calculating Inception Score...")
        generated_images = self.model.generate_samples(1024)
        inc_score = self.model_eval.compute_inception_score(generated_images)
        print(f"Inception score of GAN model: {inc_score:.4f}")

    def display_FID(self):
        print("Calculating FID...")
        generated_images = self.model.generate_samples(1024)
        FID_score = self.model_eval.compute_FID(self.model.load_data, generated_images)
        print(f"Inception score of GAN model: {FID_score:.4f}")

    def exit(self):
        print("Exiting...")
        return False

    def run(self):
        running = True
        while running:
            try:
                user_input = input("AutoDoppelGANger> ").strip()
                if user_input:
                    parts = user_input.split()
                    if parts[0] == 'help':
                        self.help()
                    elif parts[0] == 'lddst':
                        try:
                            self.load_dataset(parts[1])
                        except IndexError:
                            print("Usage: lddst <filepath>")
                    elif parts[0] == 'traingan':
                        try:
                            self.train_gan(parts[1])
                        except IndexError:
                            print("Usage: traingan <path to json file>")
                    elif parts[0] == 'trainbvae':
                        try:
                            self.train_bvae(parts[1])
                        except IndexError:
                            print("Usage: trainbvae <path to json file>")
                    elif parts[0] == 'ldwghtsgan':
                        try:
                            self.load_weights_gan(parts[1], parts[2])
                        except IndexError:
                            print("Usage: ldwghtsgan <path to discriminator weights> <path to generator weights>")
                    elif parts[0] == 'ldwghtsvae':
                        try:
                            self.load_weights_vae(parts[1])
                        except IndexError:
                            print("Usage: ldwghtsvae <path to weights>")
                    elif parts[0] == 'gensam':
                        try:
                            self.generate_samples(parts[1], parts[2], parts[3], parts[4])
                        except IndexError:
                            print("Usage: gensam <type> <num_samples> <num_rows> <num_columns>")
                    elif parts[0] == 'gendis':
                        try:
                            self.generate_disentangled_samples(parts[1], parts[2], parts[3])
                        except IndexError:
                            print("Usage: gensam <num_samples> <num_rows> <num_columns>")
                    # elif parts[0] == 'fid':
                    #    self.display_FID()
                    elif parts[0] == 'incsc':
                        self.display_inception_score()
                    elif parts[0] == 'exit':
                        break
                    else:
                        print("Unknown command. Type 'help' for help.")
                else:
                    continue
            except KeyboardInterrupt:
                print("  Program terminated by user.")
                break

