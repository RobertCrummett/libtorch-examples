#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
// #include <c10d/ProcessGroupMPI.hpp>
#include <torch/torch.h>
#include <iostream>

// Define a convolutional model
struct Model : torch::nn::Module {
	Model() : conv1(torch::nn::Conv2dOptions(1, 10, 5)),
		conv2(torch::nn::Conv2dOptions(10, 20, 5)), fc1(320, 50), fc2(50, 10)
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv2_drop", conv2_drop);
		register_module("fc1", fc1);
		register_module("fc2", fc2);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
		x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
		x = x.view({-1, 320});
		x = torch::relu(fc1->forward(x));
		x = torch::dropout(x, 0.5, is_training());
		x = fc2->forward(x);
		return torch::log_softmax(x, 1);
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Dropout2d conv2_drop;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};

void waitWork(std::shared_ptr<c10::ProcessGroupMPI> pg, std::vector<std::shared_ptr<c10::ProcessGroup::Work>> works) {
	for (auto& work : works) {
		try {
			work->wait();
		} catch (const std::exception& ex) {
			std::cerr << "Exception received: " << ex.what() << std::endl;
			pg->abort();
		}
	}
}

int main() {}
