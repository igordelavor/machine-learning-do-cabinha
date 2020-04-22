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
  var rn = new RedeNeural(2,3,2);
  let arr = [1,2];
  rn.train(arr,[0,1]);
}

function draw() {
  // put drawing code here
}