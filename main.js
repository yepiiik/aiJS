const perceptron = require('./perceptron.js');

const data_dict = {
	xxs: [[82, 62, 87]],
	xs: [[86, 68, 91]],
	s: [[90, 73, 95]],
	m: [[94, 79, 99]],
	l: [[98, 85, 103]],
	xl: [[102, 91, 107]],
	xxl: [[106, 97, 110], [110, 103, 114]],
	xxxl: [[114, 109, 118], [118, 115, 122], [122, 120, 126]],
	xxxxl: [[126, 123, 130], [130, 126, 133]],
	xxxxxl: [[134, 130, 135], [138, 135, 138]],
}

const brain = new perceptron.Brain({
	inputsCount: 3,
	hiddenLayers: [[21]],
	outputsCount: 10,
	normalizationInputs: true
});

const learning_rate = 0.01;
const epochs = 1000000;

brain.train(data_dict, learning_rate, epochs);

console.log(brain.predict([126, 123, 130]));