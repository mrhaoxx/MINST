#pragma once

#include <string_view>
#include <bit>
#include <cassert>
#include <fstream>
#include <vector>

template<typename raw_pixel, int ColN, int RowN,bool cache>
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

        if constexpr(cache){
            cache_images.resize(number_of_images);
            cache_labels.resize(number_of_images);
            imgs.read(reinterpret_cast<char*>(cache_images.data()),number_of_images * sizeof(raw_pixel) * ColN * RowN);
            labels.read(reinterpret_cast<char*>(cache_labels.data()),number_of_images * sizeof(uint8_t));
        }

        std::print("Number of images: {}\n", number_of_images);
        std::print("Rows: {}\n", rows);
        std::print("Cols: {}\n", cols);
    };


    template<int batch_size = 1>
    std::pair<std::array<std::array<raw_pixel,ColN * RowN>, batch_size>, std::array<uint8_t, batch_size>> read(){
        if constexpr(cache){
            std::array<std::array<raw_pixel,ColN * RowN>, batch_size> images;
            std::array<uint8_t, batch_size> _labels;
            for(int i = 0; (i < batch_size); i++){
                images[i] = cache_images[cur];
                _labels[i] = cache_labels[cur];
                cur++;
            }

            // std::print("Acur: {} / {}\n",cur,cache_images.size());
            return {images,_labels};
        }
        return {read_images<batch_size>(), read_labels<batch_size>()};
    }

    bool eof(){
        if constexpr(cache){
            // std::print("cur: {}\n",cur);
            return cur >= cache_images.size();
        }
        return imgs.eof() || labels.eof();
    }

    void reset(){
        if constexpr(cache){
            cur = 0;
        }else{
            imgs.clear();
            labels.clear();
            imgs.seekg(16);
            labels.seekg(8);
        }
    }

    
    ~DataLoader(){};

private:


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


    uint32_t number_of_images;

    std::vector<std::array<raw_pixel,ColN * RowN>> cache_images;
    std::vector<uint8_t> cache_labels;

    size_t cur = 0;

    std::fstream imgs;
    std::fstream labels;

};

