#pragma once

#include <string_view>
#include <bit>
#include <cassert>
#include <fstream>

template<typename raw_pixel, int ColN, int RowN>
class DataLoader {
public:
    DataLoader(std::string_view image_filename, std::string_view label_filename) {

        uint32_t magic_number,rows,cols, number_of_images_;
  
        imgs = std::fstream(image_filename.data(), std::ios::in | std::ios::binary);
        labels = std::fstream(label_filename.data(), std::ios::in | std::ios::binary);

        assert(imgs || labels);
        
        [&](auto&... args) {
            (..., imgs.read(reinterpret_cast<char*>(&args), sizeof(args)));
            if constexpr (std::endian::native == std::endian::little) 
                (..., (args = std::byteswap(args)));
        }(magic_number, number_of_images, rows, cols);


        assert(magic_number == 2051);
        assert(rows == RowN);
        assert(cols == ColN);
        
        [&](auto&... args) {
            (..., labels.read(reinterpret_cast<char*>(&args), sizeof(args)));
            if constexpr (std::endian::native == std::endian::little) 
                (..., (args = std::byteswap(args)));
        }(magic_number, number_of_images_);
        

        assert(magic_number == 2049);

        assert(number_of_images == number_of_images_);
      
        

  

    };
    template<int batch_size = 1>
    std::array<std::array<raw_pixel,ColN * RowN>, batch_size> read_images(){
        std::array<std::array<raw_pixel, ColN * RowN>,batch_size> images;
        imgs.read(reinterpret_cast<char*>(images.data()),batch_size * sizeof(raw_pixel) * ColN * RowN);

        return images;
    }

    template<int batch_size = 1>
    std::array<uint8_t, batch_size> read_labels(){
        std::array<uint8_t,batch_size> _labels;
        labels.read(reinterpret_cast<char*>(_labels.data()),batch_size * sizeof(uint8_t));

        return _labels;
    }

    bool eof(){
        return imgs.eof() || labels.eof();
    }
    
    ~DataLoader(){};

private:

    uint32_t number_of_images;

    std::fstream imgs;
    std::fstream labels;
};

