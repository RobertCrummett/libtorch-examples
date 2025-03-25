#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>

struct Options {
	int image_size = 224;
	size_t train_batch_size = 8;
	size_t test_batch_size = 200;
	size_t iterations = 10;
	size_t log_interval = 20;
	// path must end in delimiter
	std::string datasetPath = "./dataset/";
	std::string infoFilePath = "info.txt";
	torch::DeviceType device = torch::kCPU;
};

static Options options;

using Data = std::vector<std::pair<std::string, long>>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
	using Example = torch::data::Example<>;

	const Data data;

	public:
	CustomDataset(const Data& data) : data(data) {}

	Example get(size_t index) {
		std::string path = options.datasetPath + data[index].first;
		auto mat = cv::imread(path);
		assert(!mat.empty());

		cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
		std::vector<cv::Mat> channels(3);
		cv::split(mat, channels);

		auto R = torch::from_blob(
				channels[2].ptr(),
				{options.image_size, options.image_size},
				torch::kUInt8);
		auto G = torch::from_blob(
				channels[1].ptr(),
				{options.image_size, options.image_size},
				torch::kUInt8);
		auto B = torch::from_blob(
				channels[0].ptr(),
				{options.image_size, options.image_size},
				torch::kUInt8);

		auto tdata = torch::cat({R, G, B})
			.view({3, options.image_size, options.image_size})
			.to(torch::kFloat);
		auto tlabel = torch::tensor(data[index].second, torch::kLong);
		return {tdata, tlable};
	}

	torch::optional<size_t> size() const {
		return data.size();
	}
};

std::pair<Data, Data> readInfo() {
	std::random_device randomDevice;
	std::mt19937 mersenneTwisterGenerator(randomDevice());
	Data train, test;
	
	std::ifstream stream(options.infoFilePath);
	assert(stream.is_open());

	long label;
	std::string path, type;

	while (true) {
		stream >> path >> label >> type;

		if (type == "train")
			train.push_back(std::make_pair(path, label));
		else if (type == "test")
			// TODO: Continue here: https://github.com/pytorch/examples/blob/main/cpp/custom-dataset/custom-dataset.cpp
	}
}

int main() {
}
