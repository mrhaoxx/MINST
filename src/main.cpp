#include <print>

#include "image.hpp"
#include "dataloader.hpp"

#include "matrix.hpp"
#include "nn.hpp"

#include <iostream>

int main()
{
    DataLoader<uint8_t, 28, 28, true> loader("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
    DataLoader<uint8_t, 28, 28, true> loader_test("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");

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

        while (!loader.eof())
        {
            auto [imgs, labels] = loader.read<1000>();

            for (int i = 0; i < 1000; i++)
            {

                auto img = Image<uint8_t, 28, 28>(imgs[i]);
                auto p = img.reshape<1, 784>().scale(1.0 / 255.0);

                auto b = labels[i];
                auto label = nn::function::onehot<double, 1, 10>(Matrix<double, 1, 1>({double(b)}));
                l1.zero_grad();
                l2.zero_grad();

                auto act = seq.forward(p);
                auto loss = ce.forward(act, label);

                auto dloss = ce.backward(act, label);
                seq.backward(dloss);

                l1.step(0.001);
                l2.step(0.001);
       
                total_loss += loss;
                total_train++;

                // std::cout << dloss << dl1 << std::endl;
                // std::cout << img << int(b) << std::endl;
                // std::cout << l1r <<act << label;

           
            }
        }

        std::cout << "Step " << step << " Loss " << total_loss / total_train  << std::endl;

        loader_test.reset();
        int correct = 0, total = 0;

        while (!loader_test.eof())
        {
            auto [imgs, labels] = loader_test.read<100>();

            for (int i = 0; i < 100; i++)
            {

                auto img = Image<uint8_t, 28, 28>(imgs[i]);
                auto p = img.reshape<1, 784>().scale(1.0 / 255.0);

                auto b = labels[i];
                auto act = seq.forward(p);

                auto max = act.data[0];
                int max_idx = 0;
                for (int j = 1; j < 10; j++)
                {
                    if (act.data[j] > max)
                    {
                        max = act.data[j];
                        max_idx = j;
                    }
                }

                if (max_idx == b)
                {
                    correct++;
                }
                total++;
            }
        }

        
        std::cout << "verify " << correct << "/" << total << "  " << correct / float(total) << std::endl;
    }
    return 0;
}