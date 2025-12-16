/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Network
*/

#include "Network.hpp"

my_torch::Network::Network()
{
}

my_torch::Network::~Network()
{
}

my_torch::Matrix softmax(const my_torch::Matrix& input)
{
    int num_rows = input.getRows(); // N (Batch Size)
    int num_cols = input.getCols(); // O (Output Size, ici 3)

    // La matrice de résultat aura la même dimension N x O
    my_torch::Matrix result(num_rows, num_cols);

    // Itérer sur chaque exemple (ligne) du lot
    for (int r = 0; r < num_rows; ++r) {
        double sum_exp = 0.0;

        // 1. Calculer exp(x) pour tous les éléments de la ligne r et sommer
        for (int c = 0; c < num_cols; ++c) {
            // Note: Pour une meilleure stabilité numérique, il est souvent préférable de soustraire 
            // le maximum de la ligne à chaque élément avant l'exponentielle (max trick), mais 
            // nous allons garder la version simple pour éviter d'introduire trop de complexité.
            double exp_val = std::exp(input.at(r, c)); 
            result.at(r, c) = exp_val; // Stocke temporairement exp(x)
            sum_exp += exp_val;
        }

        // 2. Diviser chaque exp(x) par la somme totale (normalisation)
        for (int c = 0; c < num_cols; ++c) {
            // Applique la formule Softmax: exp(x_i) / sum(exp(x))
            result.at(r, c) /= sum_exp; 
        }
    }

    return result;
}

void my_torch::Network::init_network(const std::string loadfile, const std::string& chessfile)
{
    this->parse_untrained_nn(loadfile); // Parse the nn file
    std::vector<std::vector<double>> inputs = _parser.parse(chessfile, _matrix_output);

    // Transform (each) chess position into a (1, 64) matrix
    _matrix_input = _analyzer.vector_to_matrix(inputs);
}

void my_torch::Network::parse_untrained_nn(const std::string &blanknn)
{
    NetwokrHeader_t header;

    std::ifstream ifd(blanknn, std::ios::binary);
    ifd.seekg(0, std::ios::beg);
    ifd.read(reinterpret_cast<char *>(&header), sizeof(header));

    if (header.magicNumber != MAGIC_NUMBER) {
        std::cerr << "Invalid magic number in FEN file." << std::endl;
        return;
    }

    this->_layerCount = header.layerCount;

    for (unsigned int i = 0; i < header.layerCount; i++) {
        uint32_t nb_wandb;
        ifd.read(reinterpret_cast<char *>(&nb_wandb), sizeof(uint32_t));
        _layer_size.push_back(nb_wandb);
    }

    for (unsigned int i = 1; i < header.layerCount; i++) {
        uint32_t l1 = _layer_size.at(i-1);
        uint32_t l2 = _layer_size.at(i);
        uint32_t totalweights = l1 * l2;
        Layer layer(l1, l2);
        std::vector<double> weights;

        for (unsigned int j = 0; j < totalweights; j++) {
            float weight;
            ifd.read(reinterpret_cast<char *>(&weight), sizeof(float));
            weights.push_back(weight);
        }
        layer.setWeights(weights);
        this->layers.push_back(layer);
    }

    for (unsigned int i = 1; i < header.layerCount; i++) {
        uint32_t nbbiases = _layer_size.at(i);
        std::vector<double> biases;
        for (unsigned int j = 0; j < nbbiases; j++) {
            float biase;
            ifd.read(reinterpret_cast<char *>(&biase), sizeof(float));
            biases.push_back(biase);
        }
        this->layers.at(i-1).setBiases(biases);
    }
}

void my_torch::Network::pack_trained_nn(const std::string& trainfile)
{
    std::FILE* f = std::fopen(trainfile.c_str(), "wb");

    if (!f) {
        std::cerr << "File couldn't be opened !" << std::endl;
        return;
    }

    // Write header (magic number and layer count)
    uint32_t magic = MAGIC_NUMBER;
    uint32_t layerCount = this->_layerCount;
    std::fwrite(&magic, sizeof(uint32_t), 1, f);
    std::fwrite(&layerCount, sizeof(uint32_t), 1, f);

    // Write layer sizes (topology)
    for (unsigned int i = 0; i < layerCount; i++) {
        uint32_t layerSize = this->_layer_size.at(i);
        std::fwrite(&layerSize, sizeof(uint32_t), 1, f);
    }

    // Write all weights
    for (unsigned int i = 1; i < layerCount; i++) {
        Matrix weights = this->layers.at(i-1).getWeights();
        for (int j = 0; j < weights.getRows(); j++) {
            for (int k = 0; k < weights.getCols(); k++) {
                float weight = static_cast<float>(weights.at(j, k));
                std::fwrite(&weight, sizeof(float), 1, f);
            }
        }
    }

    // Write all biases
    for (unsigned int i = 1; i < layerCount; i++) {
        Matrix biases = this->layers.at(i-1).getBiases();
        for (int j = 0; j < biases.getRows(); j++) {
            for (int k = 0; k < biases.getCols(); k++) {
                float bias = static_cast<float>(biases.at(j, k));
                std::fwrite(&bias, sizeof(float), 1, f);
            }
        }
    }

    std::fclose(f);
}

my_torch::Matrix my_torch::Network::forward(Matrix input)
{
    Matrix current = input;

    for (size_t i = 0; i < layers.size(); ++i) {
        current = layers[i].forward(current); // Propagation normale

        if (i == layers.size() - 1 && current.getCols() == 3) {
            current = softmax(current);
        }
    }
    return current;
}

void my_torch::Network::train(double learning_rate, const std::string& savefile)
{
    std::size_t num_examples = _matrix_input.size();
    if (num_examples == 0) {
        std::cerr << "Training failed: No input data loaded." << std::endl;
        return;
    }
    std::size_t num_batches = (num_examples + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int epoch = 0; epoch < 50; epoch++) {
        double total_loss = 0.0;

        for (std::size_t b = 0; b < num_batches; b++) {

            std::size_t start_index = b * BATCH_SIZE;
            std::size_t end_index = std::min(start_index + BATCH_SIZE, num_examples);
            std::size_t current_batch_size = end_index - start_index;

            // 1. CREATION DES MATRICES DE LOT (N x 768 et N x 3)
            my_torch::Matrix batch_input(current_batch_size, INPUTS, false);
            my_torch::Matrix batch_expected_output(current_batch_size, 3, false);

            for (std::size_t i = 0; i < current_batch_size; ++i) {
                // Copie des entrées et sorties dans les matrices N x M
                for (int c = 0; c < INPUTS; ++c) {
                    batch_input.at(i, c) = _matrix_input[start_index + i].at(0, c);
                }
                for (int c = 0; c < 3; ++c) { // 3 est la taille de sortie (Nothing, Check, Checkmate)
                    batch_expected_output.at(i, c) = _matrix_output[start_index + i].at(0, c);
                }
            }

            // 2. Propagation avant (Forward Pass) pour tout le lot
            Matrix output = this->forward(batch_input);

            double batch_loss = 0.0;
            for (std::size_t i = 0; i < current_batch_size; ++i) {
                double example_loss = 0.0;
                for (int c = 0; c < 3; ++c) {
                    double y = batch_expected_output.at(i, c);
                    double hat_y = output.at(i, c);

                    if (y > 0.0 && hat_y > 1e-12) {
                        example_loss += y * std::log(hat_y);
                    }
                }
                batch_loss += -example_loss;
            }
            total_loss += batch_loss;

            // 4. Calcul du Gradient initial (dL/dZ_last)
            // Gradient initial = ^y - y (N x 3)
            Matrix gradient = output - batch_expected_output;

            // Le gradient doit être moyenné par la taille du lot
            Matrix averaged_gradient = gradient * (1.0 / current_batch_size);

            // 5. Rétropropagation (Backward Pass)
            this->backward(averaged_gradient, learning_rate);
        }

        // Affichage de la perte moyenne de l'époque
        std::cout << "Epoch " << epoch + 1 << " Loss: " << total_loss / num_examples << std::endl;
    }
    this->pack_trained_nn(savefile);
}

void my_torch::Network::predict()
{
    for (std::size_t i = 0; i < _matrix_input.size(); i++) {
        my_torch::Matrix output = this->forward(_matrix_input[i]);

        if (output.getCols() < 3) {
            std::cerr << "Prediction output error: Expected 3 classes, got " << output.getCols() << std::endl;
            return;
        }

        double max_prob = -1.0;
        int predicted_index = -1;

        for (int j = 0; j < 3; ++j) {
            if (output.at(0, j) > max_prob) {
                max_prob = output.at(0, j);
                predicted_index = j;
            }
        }

        if (predicted_index == 0) {
            std::cout << "Nothing" << std::endl;
        } else if (predicted_index == 1) {
            std::cout << "Check" << std::endl;
        } else if (predicted_index == 2) {
            std::cout << "Checkmate" << std::endl;
        } else {
            std::cout << "Prediction Error (Index " << predicted_index << ")" << std::endl;
        }
    }
}

void my_torch::Network::backward(Matrix& gradient, double learning_rate)
{
    Matrix current_gradient = gradient;

    for (int i = layers.size() - 1; i >= 0; --i) {
        current_gradient = layers[i].backward(current_gradient, learning_rate);
    }
}