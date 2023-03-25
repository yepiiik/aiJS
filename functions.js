// Функция активации сигмоид
function sigmoid(x) {
	return 1 / (1 + Math.exp(-x));
}

// Производная функции сигмоид
function dxSigmoid(x) {
	var fx = sigmoid(x);
	return fx * (1 - fx);
}

// Функция траниспонирования матриц
function transponse(matrix) {
	return matrix[0].map((col, i) => matrix.map(row => row[i]));
}

// Функиция разности матриц
function matrixDifference(xMatrix, yMatrix) {
	if (xMatrix.length == yMatrix.length && transponse([xMatrix]).length == transponse([yMatrix]).length) {
		var zMatrix = [];
		for (var i = 0; i < xMatrix.length; i++) {
			zMatrix.push(xMatrix[i] - yMatrix[i]);
		}
		return zMatrix;
	}
}

// Функция средней квадратичной ошибки
function mseLoss(y_true, y_pred) {
	return 1 / y_true.length * sumArr(matrixDifference(y_true, y_pred));
}

function mse(y_true, y_pred) {
	return avergeArr(matrixDifference(y_true, y_pred)) ** 2;
}

// Функция сумы всех элементов масива
function sumArr(arr) {
	var sumArr = 0;
	for (var i = 0; i < arr.length; i++) {
		sumArr += arr[i];
	}
	return sumArr;
}

// Функция среднего в масиве
function avergeArr(arr) {
	return sumArr(arr) / arr.length;
}

function prepareData(obj) {
	const count = Object.keys(obj).length;
	let inputs = [];
	let outputs = [];
	for (var key in obj) {
		for (var i = 0; i < obj[key].length; i++) {
			inputs.push(obj[key][i]);
			var arr = new Array(count).fill(0, 0, count)
			arr[Object.keys(obj).indexOf(key)] = 1
			outputs.push(arr)
		}
	}
	return [inputs, outputs];
}

function stdNorm(vector) {
	let result_vector = [];

	let sum = sumArr(vector)

	for (var i in vector) {
		result_vector.push(vector[i] / sum);
	}
	return result_vector;
}

module.exports = {
	sigmoid: sigmoid,
	dxSigmoid: dxSigmoid,
	transponse: transponse,
	matrixDifference: matrixDifference,
	mseLoss: mseLoss,
	mse: mse,
	sumArr: sumArr,
	avergeArr: avergeArr,
	prepareData: prepareData,
	stdNorm: stdNorm,
}