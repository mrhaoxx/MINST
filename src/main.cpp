#include <print>

#include "image.hpp"
#include "dataloader.hpp"

#include "matrix.hpp"
#include "nn.hpp"

#include <iostream>

int main()
{
    DataLoader<uint8_t, 28, 28> loader("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");

    DataLoader<uint8_t, 28, 28> loader_test("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");

    srand(0);
    Linear l1(random<double, 784, 10>(), random<double, 1, 10>());
    Softmax<double, 1, 10> s1;
    CrossEntropy<double, 1, 10> ce;

    while (!loader.eof())
    {
        auto imgs = loader.read_images<1000>();
        auto labels = loader.read_labels<1000>();

        double total_loss = 0;

        for (int i = 0; i < 1000; i++)
        {

            auto img = Image<uint8_t, 28, 28>(imgs[i]);
            auto p = img.reshape<1, 784>().scale(1.0 / 255.0);

            auto b = labels[i];
            auto label = nn::function::onehot<double, 1, 10>(Matrix<double, 1, 1>({double(b)}));
            l1.zero_grad();


            auto l1r = l1.forward(p);
            auto act = s1.forward(l1r);
            auto loss = ce.forward(act, label);

            auto dloss = ce.backward(act, label);
            auto ds1 = s1.backward(l1r, dloss);
            auto dl1 = l1.backward(p, ds1);

            // std::cout << img << int(b) << std::endl;
            // std::cout << l1r <<act << label;
  
            total_loss += loss;
            // if (std::isnan(loss))
            // {
            //     break;
            // }

            // std::cout << dloss << dl1 << std::endl;

            l1.step(0.005);
        }
        std::cout << total_loss / 1000 << std::endl;

    }

    int correct = 0, total = 0;

    while (!loader_test.eof())
    {
        auto imgs = loader_test.read_images<1000>();
        auto labels = loader_test.read_labels<1000>();


        for (int i = 0; i < 1000; i++)
        {

            auto img = Image<uint8_t, 28, 28>(imgs[i]);
            auto p = img.reshape<1, 784>().scale(1.0 / 255.0);

            auto b = labels[i];
            auto label = nn::function::onehot<double, 1, 10>(Matrix<double, 1, 1>({double(b)}));

            auto l1r = l1.forward(p);
            auto act = s1.forward(l1r);
            auto loss = ce.forward(act, label);
            
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
            total ++;
        }
    }
    
    std::cout <<  correct  << " " <<  total << "  " << correct / float(total) << std::endl;

    return 0;
}