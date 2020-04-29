function setup() {
  // put setup code here
  createCanvas(500, 500);
  background(0);
  
  //var m1 = new Matrix(1,2);
  //var m2 = new Matrix(2,1);
  //console.log(m1.data);
  //console.log(m2.data);
  //console.log(Matrix.multiply(m1,m2).data);
  // console.log(m);
  //var rn = new RedeNeural(1,3,1); //1 entrada, 3 hidden e 1 saida
  //var arr = [1,2];
  //Matrix.arrayToMatrix(arr);
  //rn.feedforward(arr);
  //var rn = new RedeNeural(2,3,2);
  //let arr = [1,2];
  //rn.train(arr,[0,1]);
  rn = new RedeNeural(2,3,1);
  var train = true;
  // XOR Problem
  dataset = {
    inputs:
        [[1, 1],
        [1, 0],
        [0, 1],
        [0, 0]],
    outputs:
        [[0],
        [1],
        [1],
        [0]]
  }
}

function draw() {
    if (train) {
    for (var i = 0; i < 10000; i++) {
        var index = floor(random(4));
        nn.train(dataset.inputs[index], dataset.outputs[index]);
    }
    if (nn.predict([0, 0])[0] < 0.04 && nn.predict([1, 0])[0] > 0.98) {
        train = false;
        console.log("terminou");
    }
  }
}
