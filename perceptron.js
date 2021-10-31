// Функция активации сигмоид
function sigmoid(x) {
	return 1 / (1 + Math.exp(-x));
}

// Походная функции сигмоид
function dx_sigmoid(x) {
	var fx = sigmoid(x);
	return fx * (1 - fx);
}

// Функция траниспонирования матриц
function transponse(matrix) {
	return matrix[0].map((col, i) => matrix.map(row => row[i]));
}

// Функиция разности матриц
function matrix_difference(xMatrix, yMatrix) {
	if (xMatrix.length == yMatrix.length && transponse([xMatrix]).length == transponse([yMatrix]).length) {
		var zMatrix = [];
		for (var i = 0; i < xMatrix.length; i++) {
			zMatrix.push(xMatrix[i] - yMatrix[i]);
		}
		return zMatrix;
	}
}

// Функция средней квадратичной ошибки
function mse_loss(y_true, y_pred) {
	return 1 / y_true.length * sumArr(matrix_difference(y_true, y_pred));
}

function MSE(y_true, y_pred) {
	return averge(matrix_difference(y_true, y_pred)) ** 2;
}

// Функция сумы всех элементов масива
function sumArr(arr) {
	var sumArr = 0;
	for (var i = 0; i < arr.length; i++) {
		sumArr += arr[i];
	}
	return sumArr;
}

// Function averge
function averge(arr) {
	return sumArr(arr) / arr.length;
}

/*
------------------------ Классы ---------------------
*/

class Normalization {
	constructor(xMin, xMax, d1, d2) {
		this.xMin = xMin;
		this.xMax = xMax;
		this.d1 = d1;
		this.d2 = d2;
	}

	start(x) {
		return ((x - this.xMin)*(this.d2 - this.d1)) / (this.xMax - this.xMin) + this.d1;
	}
}

class Neuron {
	constructor(weights) {
		this.weights = weights;
	}

	feedForward(inputs) {
		// скалярное произведение векторов
		var total = 0;
		for (var i = 0; i < inputs.length; i++) {
			total += inputs[i] * this.weights[i];
		}

		// возращаем результат функции активации
		return sigmoid(total);
	}
}

class Brain {
	constructor(options) {
		this.inputsCount = options.inputsCount;
		this.hiddenLayers = options.hiddenLayers;
		this.outputsCount = options.outputsCount;
		this.normalizationInputs = options.normalizationInputs;

		this.initNeurons();
	}

	feedForward(x) {
		this.neuronsResults = [];
		var out = [];
		var layerOut = x;
		for (var layer = 0; layer < this.neurons.length; layer++) {
			var actualLayer = this.neurons[layer];
			for (var i = 0; i < actualLayer.length; i++) {
				out.push(actualLayer[i].feedForward(layerOut));
			}
			this.neuronsResults.push(out);
			layerOut = out;
			out = [];
		}
		return layerOut;
	}

	initWeight(inputsCount) {
		var weights = [];

		// создаем нужное количество весов
		for (var w = 0; w < inputsCount; w++){
			weights.push(Math.random());
		}

		return weights;
	}

	initNeurons() {
		this.neurons = [];
		var actualLayer = [];
		var preventCount = this.inputsCount;
		for (var layers = 0; layers < this.hiddenLayers.length; layers++) {
			for (var i = 0; i < this.hiddenLayers[layers]; i++) {
				actualLayer.push(new Neuron(this.initWeight(preventCount)));
			}
			preventCount = actualLayer.length
			this.neurons.push(actualLayer);
			actualLayer = [];

		}

		for (var i = 0; i < this.outputsCount; i++) {
			actualLayer.push(new Neuron(this.initWeight(preventCount)));
		}
		this.neurons.push(actualLayer);
		actualLayer = [];
	}

	train(inputs, outputs, learning_rate, epochs) {

		// Errors -----------------------------------

		if (inputs.length == 1) {
			console.log(`Error: 001 (Training examples must be more than 1)`);
			return;
		}

		// Normalization on / off -----------------------------------

		if (this.normalizationInputs == true) {
			var inputs = this.normalization_train(inputs, 0, 1);
			console.log(inputs)
		}

		// Epochs -----------------------------------

		while (epochs != 0) {
			for (var example = 0; example < inputs.length; example++) {
				var train_example_inputs = inputs[example];
				var train_example_outputs = outputs[example];
				var actual_out;
				this.feedForward(train_example_inputs);


				// Creating empty arrays -----------------------------------

				var weight_delta_arr = [];
				var error_arr = [];
				for (var layer = 0; layer < this.neurons.length; layer++) {
					weight_delta_arr.push([]);
					error_arr.push([]);
					for (var i = 0; i < this.neurons[layer].length; i++) {
						weight_delta_arr[layer].push([]);
						error_arr[layer].push([]);
					}
				}

				// Getting neuron's error & weight delta -----------------------------------

				var error = 0;
				for (var layer = this.neurons.length - 1; layer != -1; layer--) {
					for (var i = 0; i < this.neurons[layer].length; i++) {
						actual_out = this.neuronsResults[layer][i];
						if (layer == this.neurons.length - 1) {
							error = actual_out - train_example_outputs[i];
						} else {
							for (var layer_neurons = 0; layer_neurons < this.neurons[layer+1].length; layer_neurons++) {
								error += this.neurons[layer+1][layer_neurons].weights[i] * weight_delta_arr[layer+1][layer_neurons];
							}
						}
						error_arr[layer][i] = error;
						weight_delta_arr[layer][i] = error * (actual_out * (1 - actual_out));
						error = 0;
					}
				}

				// Back propagation -----------------------------------

				for (var layer = this.neurons.length - 1; layer != -1; layer--) {
					for (var i = 0; i < this.neurons[layer].length; i++) {
						for (var weight_num = 0; weight_num < this.neurons[layer][i].weights.length; weight_num++) {
							if (layer - 1 != -1) {
								this.neurons[layer][i].weights[weight_num] = this.neurons[layer][i].weights[weight_num] - this.neuronsResults[layer - 1][weight_num] * weight_delta_arr[layer][i] * learning_rate;
							} else {
								this.neurons[layer][i].weights[weight_num] = this.neurons[layer][i].weights[weight_num] - train_example_inputs[weight_num] * weight_delta_arr[layer][i] * learning_rate;
							}
						}
					}
				}
			}

		epochs --
		}
	}

	normalization_train(inputs, d1, d2) {
		this.arrange_inputs = [];
		var transponse_inputs = transponse(inputs);
		var normalized_inputs = [];
		var max;
		var min;
		var x;

		for (var i = 0; i < transponse_inputs.length; i++) {
			max = Math.max.apply(null, transponse_inputs[i]);
			min = Math.min.apply(null, transponse_inputs[i])
			this.arrange_inputs[i] = new Normalization(min, max, d1, d2);
		}

		console.log(this.arrange_inputs);

		for (var i = 0; i < transponse_inputs.length; i++) {
			normalized_inputs.push([]);
			for (var value_count = 0; value_count < transponse_inputs[i].length; value_count++) {
				x = transponse_inputs[i][value_count];
				min = this.arrange_inputs[i].xMin;
				max = this.arrange_inputs[i].xMax;
				normalized_inputs[i][value_count] = ((x - min)*(d2 - d1)) / (max - min) + d1;
			}
		}
		return transponse(normalized_inputs);
	}

	normalization(inputs) {
		var transponse_inputs = transponse(inputs);
		var normalized_inputs = [];
		var max;
		var min;
		var x;
		var d1;
		var d2;
		
		for (var i = 0; i < transponse_inputs.length; i++) {
			normalized_inputs.push([]);
			for (var value_count = 0; value_count < transponse_inputs[i].length; value_count++) {
				x = transponse_inputs[i][value_count];
				min = this.arrange_inputs[i].xMin;
				max = this.arrange_inputs[i].xMax;
				d1 = this.arrange_inputs[i].d1;
				d2 = this.arrange_inputs[i].d2;
				normalized_inputs[i][value_count] = ((x - min)*(d2 - d1)) / (max - min) + d1;
			}
		}
		return transponse(normalized_inputs);
	}

	predict(inputs) {
		// Normaliztion on / off -----------------------------------

		if (this.normalizationInputs == true) {
			var inputs = this.normalization([inputs]);
		}
		var result = this.feedForward(inputs[0])
		return result;
	}
}

const inputs = [
	[82, 62, 87],
	[86, 68, 91],
	[90, 73, 95],
	[94, 79, 99],
	[98, 85, 103],
	[102, 91, 107],
	[106, 97, 110],
	[110, 103, 114],
	[114, 109, 118],
	[118, 115, 122],
	[122, 120, 126],
	[126, 123, 130],
	[130, 126, 133],
	[134, 130, 135],
	[138, 135, 138]
];

const outputs = [
	[1,0,0,0,0,0,0,0,0,0],
	[0,1,0,0,0,0,0,0,0,0],
	[0,0,1,0,0,0,0,0,0,0],
	[0,0,0,1,0,0,0,0,0,0],
	[0,0,0,0,1,0,0,0,0,0],
	[0,0,0,0,0,1,0,0,0,0],
	[0,0,0,0,0,0,1,0,0,0],
	[0,0,0,0,0,0,1,0,0,0],
	[0,0,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,0,0,1,0],
	[0,0,0,0,0,0,0,0,1,0],
	[0,0,0,0,0,0,0,0,0,1],
	[0,0,0,0,0,0,0,0,0,1],
];

const brain = new Brain({
	inputsCount: 3,
	hiddenLayers: [[6],[6]],
	outputsCount: 10,
	// normalizationInputs: true
});

const learning_rate = 0.01;
const epochs = 100000;

brain.train(inputs, outputs, learning_rate, epochs);

console.log(brain.predict([122, 122, 115]));