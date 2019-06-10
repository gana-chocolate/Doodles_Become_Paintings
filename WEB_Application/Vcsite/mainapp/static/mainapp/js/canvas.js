var pos = {
  drawable : false,
  x: -1,
  y: -1
};
var canvas, ctx;
var canvas2, ctx2;
var canvas3, ctx3;
var canvas4, ctx4;
function initDraw(event) {
  ctx.beginPath();
  pos.drawable = true;
  var coors = getPosition(event);
  pos.X = coors.X;
  pos.Y = coors.Y;
  ctx.moveTo(pos.X, pos.Y);
}
function draw(event){
  var coors = getPosition(event);
  ctx.lineTo(coors.X, coors.Y);
  pos.X = coors.X;
  pos.Y = coors.Y;
  ctx.stroke();
}
function finishDraw(){
  pos.drawable = false;
  pos.X = -1;
  pos.Y = -1;
}
function getPosition(event) {
  var x = event.pageX - canvas.offsetLeft;
  var y = event.pageY - canvas.offsetTop;
  return {X: x, Y: y};
}


function initDraw2(event) {
  ctx2.beginPath();
  pos.drawable = true;
  var coors = getPosition2(event);
  pos.X = coors.X;
  pos.Y = coors.Y;
  ctx2.moveTo(pos.X, pos.Y);
}
function draw2(event){
  var coors = getPosition2(event);
  ctx2.lineTo(coors.X, coors.Y);
  pos.X = coors.X;
  pos.Y = coors.Y;
  ctx2.stroke();
}
function finishDraw2(){
  pos.drawable = false;
  pos.X = -1;
  pos.Y = -1;
}
function getPosition2(event) {
  var x = event.pageX - canvas2.offsetLeft;
  var y = event.pageY - canvas2.offsetTop;
  return {X: x, Y: y};
}

function initDraw3(event) {
  ctx3.beginPath();
  pos.drawable = true;
  var coors = getPosition3(event);
  pos.X = coors.X;
  pos.Y = coors.Y;
  ctx3.moveTo(pos.X, pos.Y);
}
function draw3(event){
  var coors = getPosition3(event);
  ctx3.lineTo(coors.X, coors.Y);
  pos.X = coors.X;
  pos.Y = coors.Y;
  ctx3.stroke();
}
function finishDraw3(){
  pos.drawable = false;
  pos.X = -1;
  pos.Y = -1;
}
function getPosition3(event) {
  var x = event.pageX - canvas3.offsetLeft;
  var y = event.pageY - canvas3.offsetTop;
  return {X: x, Y: y};
}

function initDraw4(event) {
  ctx4.beginPath();
  pos.drawable = true;
  var coors = getPosition4(event);
  pos.X = coors.X;
  pos.Y = coors.Y;
  ctx4.moveTo(pos.X, pos.Y);
}
function draw4(event){
  var coors = getPosition4(event);
  ctx4.lineTo(coors.X, coors.Y);
  pos.X = coors.X;
  pos.Y = coors.Y;
  ctx4.stroke();
}
function finishDraw4(){
  pos.drawable = false;
  pos.X = -1;
  pos.Y = -1;
}
function getPosition4(event) {
  var x = event.pageX - canvas4.offsetLeft;
  var y = event.pageY - canvas4.offsetTop;
  return {X: x, Y: y};
}
window.onload = function(){
  canvas = document.getElementById("canvas");
  ctx = canvas.getContext("2d");
  canvas.addEventListener("mousedown", listener);
  canvas.addEventListener("mousemove", listener);
  canvas.addEventListener("mouseup", listener);
  canvas.addEventListener("mouseout", listener);

  canvas2 = document.getElementById("canvas2");
  ctx2 = canvas2.getContext("2d");
  canvas2.addEventListener("mousedown", listener2);
  canvas2.addEventListener("mousemove", listener2);
  canvas2.addEventListener("mouseup", listener2);
  canvas2.addEventListener("mouseout", listener2);
}

function listener(event){
  switch(event.type){
      case "mousedown":
        initDraw(event);
        break;
      case "mousemove":
        if(pos.drawable)
          draw(event);
        break;
      case "mouseout":
      case "mouseup":
        finishDraw();
        break;
  }
}
function listener2(event){
  switch(event.type){
      case "mousedown":
        initDraw2(event);
        break;
      case "mousemove":
        if(pos.drawable)
          draw2(event);
        break;
      case "mouseout":
      case "mouseup":
        finishDraw2();
        break;
  }
}
function listener3(event){
  switch(event.type){
      case "mousedown":
        initDraw3(event);
        break;
      case "mousemove":
        if(pos.drawable)
          draw3(event);
        break;
      case "mouseout":
      case "mouseup":
        finishDraw3();
        break;
  }
}
function listener4(event){
  switch(event.type){
      case "mousedown":
        initDraw4(event);
        break;
      case "mousemove":
        if(pos.drawable)
          draw4(event);
        break;
      case "mouseout":
      case "mouseup":
        finishDraw4();
        break;
  }
}



function uploadCanvasData()
{
    var dataUrl = canvas.toDataURL();
    var blob = dataURItoBlob(dataUrl);
    var formData = new FormData();
    formData.append("file", blob);
    alert(dataUrl);
    console.log(formData);

    /*var request = new XMLHttpRequest();
    request.onload = completeRequest;
    request.open("POST", "mainapp:send");
    request.send(formData);*/
}

function dataURItoBlob(dataURI)
{
    var byteString = atob(dataURI.split(',')[1]);
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0]
    var ab = new ArrayBuffer(byteString.length);
    var ia = new Uint8Array(ab);
    for (var i = 0; i < byteString.length; i++)
    {
        ia[i] = byteString.charCodeAt(i);
    }

    var bb = new Blob([ab], { "type": mimeString });
    return bb;
}
/*function send(){
    var canvas = document.getElementById('image');
    var dataURL = canvas.toDataURL();
    $.ajax({
        type: "POST",
        url: "mainapp:upload ",
        data: {
            imgBase64: dataURL
        }
    }).done(function(o) {
        console.log('saved');
    });
}
function sendBase64Img() {
    var canvas2 = document.getElementById('canvas2');
    var dataURL = canvas2.getContext("2d");
    var image = new Image();
    image.src = canvas.toDataURL();

    //var dataURL = canvas2.toDataURL();//이미지 데이터가 base64 문자열로 인코딩된 데이터
    // base64문자열의 첫 부분에 위치한 'https://t1.daumcdn.net/cfile/tistory/24343B4956E6601629"");
    $.ajax({
      type: "POST",
      url: "saveBase64.jsp",
      contentType: "application/x-www-form-urlencoded; charset=utf-8",
      data: { "imgBase64": dataURL }
    }).success(function(o) {
      alert('선택영역을 서버의 이미지 파일에 저장했습니다');
    });
}*/
