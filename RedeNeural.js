function sigmoid(x){
	let result = 0;
	result += 1/(1+Math.exp(-x));
	return result;
}

function d_sigmoid(x){
	return x*(1-x);
}

class RedeNeural{
	constructor(i_nodes, h_nodes, o_nodes){
		this.i_nodes = i_nodes;
		this.h_nodes = h_nodes;
		this.o_nodes = o_nodes;
		
		this.bias_ih = new Matrix(this.h_nodes, 1);
		this.bias_ih.randomize();
		this.bias_ho = new Matrix(this.o_nodes, 1);
		this.bias_ho.randomize();
		
		this.weigths_ih = new Matrix(this.h_nodes, this.i_nodes);
		this.weigths_ih.randomize()
		
		this.weigths_ho = new Matrix(this.o_nodes, this.h_nodes)
		this.weigths_ho.randomize()

		this.learning_rate = 0.1;
		
		//this.weight_ho.print();
		//this.weight_ih.print();
	}
	
	//feedforward(arr){

		//INPUT -> HIDDEN
	//	let input = Matrix.arrayToMatrix(arr);
	//	let hidden = Matrix.multiply(this.weigths_ih, input);
	//	
	//	hidden = Matrix.add(hidden,this.bias_ih);
	//	
	//	hidden.map(sigmoid);
	//	
	//	//HIDDEN -> OUTPUT
	//	let output = Matrix.multiply(this.weigths_ho, hidden);
	//	output = Matrix.add(output,this.bias_ho);
	//	output.map(sigmoid);
	//	output.print();
	//}
	train(arr, target){
		let input = Matrix.arrayToMatrix(arr);
		let hidden = Matrix.multiply(this.weigths_ih, input);
		
		hidden = Matrix.add(hidden,this.bias_ih);
		
		hidden.map(sigmoid);
		
		//HIDDEN -> OUTPUT
		let output = Matrix.multiply(this.weigths_ho, hidden);
		output = Matrix.add(output,this.bias_ho);
		output.map(sigmoid);
		//output.print();

		// BACKPROPAGATION

		// OUTPUT -> HIDDEN
		let expected = Matrix.arrayToMatrix(target);
		let output_error = Matrix.subtract(expected, output);
		let d_output = Matrix.map(output,d_sigmoid);
		let hidden_T = Matrix.transpose(hidden);

		let gradient = Matrix.hadamard(d_output, output_error);
		gradient = Matrix.escalar_multiply(gradient, this.learning_rate);

		this.bias_ho = Matrix.add(this.bias_ho, gradient);
		let weigths_ho_deltas = Matrix.multiply(gradient, hidden_T);
		//weigths_ho_deltas.print();
		//this.weigths_ho.print();
		this.weigths_ho = Matrix.add(this.weigths_ho, weigths_ho_deltas);
		//this.weigths_ho.print();


		// HIDDEN -> INPUT
		let weigths_ho_T = Matrix.transpose(this.weigths_ho);
		let hidden_error = Matrix.multiply(weigths_ho_T, output_error);
		let d_hidden = Matrix.map(hidden_error, d_sigmoid);
		let input_T = Matrix.transpose(input);

		let gradient_H = Matrix.hadamard(hidden_error, d_hidden);
		gradient_H = Matrix.escalar_multiply(gradient, this.learning_rate);

		this.bias_ih = Matrix.add(this.bias_ih, gradient_H);

		let weigths_ih_deltas = Matrix.multiply(gradient, input_T);
		//weigths_ih_deltas.print();
		//this.weigths_ih.print();
		this.weigths_ih = Matrix.add(this.weigths_ih, weigths_ih_deltas);


	}

	predict(arr){
		let input = Matrix.arrayToMatrix(arr);
		let hidden = Matrix.multiply(this.weigths_ih, input);
		
		hidden = Matrix.add(hidden,this.bias_ih);
		
		hidden.map(sigmoid);
		
		//HIDDEN -> OUTPUT
		let output = Matrix.multiply(this.weigths_ho, hidden);
		output = Matrix.add(output,this.bias_ho);
		output.map(sigmoid);
		output = Matrix.MatrixToArray(output);
		return output;
	}
}