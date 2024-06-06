#include <print>

#include "image.hpp"
#include "dataloader.hpp"

#include "matrix.hpp"
#include "nn.hpp"

#include <iostream>
#include <vector>

int main()
{
    DataLoader<uint8_t, 28, 28, true> loader("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
    DataLoader<uint8_t, 28, 28, true> loader_test("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");

    std::vector<Matrix<double, 1, 784>> images_train;
    std::vector<Matrix<double, 1, 10>> labels_train;

    std::vector<Matrix<double, 1, 784>> images_test;
    std::vector<uint8_t> labels_test;

    while (!loader.eof())
    {
        auto [imgs, labels] = loader.read<1000>();
        for (int i = 0; i < 1000; i++)
        {
            images_train.push_back(Image<uint8_t, 28, 28>(imgs[i]).reshape<1, 784>().scale(1.0 / 255.0));
            labels_train.push_back(nn::function::onehot<double, 1, 10>(Matrix<double, 1, 1>({double(labels[i])})));
        }
    }

    while (!loader_test.eof())
    {
        auto [imgs, labels] = loader_test.read<1000>();
        for (int i = 0; i < 1000; i++)
        {
            images_test.push_back(Image<uint8_t, 28, 28>(imgs[i]).reshape<1, 784>().scale(1.0 / 255.0));
            labels_test.push_back(labels[i]);
        }
    }

    std::print("Loaded {} images\n", images_train.size());
    srand(42);

    Linear l1(random<double, 784, 128>(), random<double, 1, 128>());
    ReLU<double, 1, 128> r1;
    Dropout<double, 1, 128> d1(20);
    Linear l2(random<double, 128, 10>(), random<double, 1, 10>());
    Softmax<double, 1, 10> s1;

    CrossEntropy<double, 1, 10> ce;

    Sequential seq(l1, r1, d1, l2, s1);

    for (int step = 0; step < 100; step++)
    {
        loader.reset();

        double total_loss = 0;
        int total_train = 0;

        for (int i = 0; i < images_train.size(); i++)
        {

            auto image = images_train[i];
            auto label = labels_train[i];
            l1.zero_grad();
            l2.zero_grad();

            auto act = seq.forward(image);
            auto loss = ce.forward(act, label);

            seq.backward(ce.backward(act, label));

            l1.step(0.001);
            l2.step(0.001);

            total_loss += loss;
            total_train++;

            // std::cout << dloss << dl1 << std::endl;
            // std::cout << img << int(b) << std::endl;
            // std::cout << l1r <<act << label;
        }

        std::cout << "Step " << step << " Loss " << total_loss / total_train << std::endl;

        loader_test.reset();
        int correct = 0, total = 0;

        for (int i = 0; i < images_test.size(); i++)
        {
            auto image = images_test[i];
            auto label = labels_test[i];

            auto act = seq.forward(image);

            auto max_idx = std::distance(act.data.begin(), std::max_element(act.data.begin(), act.data.end()));

            if (max_idx == label)
            {
                correct++;
            }
            total++;
        }

        std::cout << "verify " << correct << "/" << total << "  " << correct / float(total) << std::endl;
    }
    return 0;
}