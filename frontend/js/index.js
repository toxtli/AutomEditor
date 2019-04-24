// Here goes the python service URL
var serverURL = 'data/test.json';
//var serverURL = '//localhost:5000/uploader';
var URL = window.URL || window.webkitURL;

function playSelectedFile(event) {
  var file = this.files[0];
  var fileURL = URL.createObjectURL(file);
  videoNode.src = fileURL;
  console.log(videoNode);
}

function deleteTimeFrames() {
  videoBar.innerHTML = '';
}

function createTimeFrames(slots) {
  var html = '';
  var width = (480 / slots).toFixed(2);
  for (var i = 0; i < slots; i++) {
    html += '<div id="frame_' + i + '" class="frame" style="width:' + width + 'px;"></div>'
  }
  videoBar.innerHTML = html;
}

function toggleTimeFrame(slotId) {
  var frame = document.getElementById('frame_' + slotId);
  if (frame) {
    frame.classList.toggle("selectedFrame");
  }
}

function updateKeyFrame() {
  var slotId = document.getElementById('segment').value;
  if (slotId <= videoLength) {
    videoNode.currentTime = slotId;
    toggleTimeFrame(slotId);
  }
}

function callServer() {
  $('#status').show();
  $('#status').html('Processing... <img src="img/load.gif">');
  var fd = new FormData($('#fileinfo')[0]);
  $.ajax({
    url: serverURL,  
    type: 'POST',
    dataType: "json",
    data: fd,
    success:function(data){
      console.log(data);
      for (range of data) {
        for (value of range) {
          toggleTimeFrame(value);
        }
      }
      $('#status').text('Operation completed.');
    },
    error:function(){
      $('#status').text('Error processing the file.');
    },
    contentType: false,
    processData: false 
  });
}

var videoLength = 0;
var videoBar = document.getElementById('videobar');
var inputNode = document.getElementById('filename');
var videoNode = document.getElementById('videoControl');
inputNode.addEventListener('change', playSelectedFile, false);
videoNode.addEventListener('loadedmetadata', function() {
  $('#videobar').show()
  //$('#toolBar').show()
  $('#processButton').show()
  videoLength = parseInt(videoNode.duration);
  deleteTimeFrames();
  createTimeFrames(videoLength);
});
document.getElementById('setButton').addEventListener('click', updateKeyFrame);
document.getElementById('segment').addEventListener('keyup', function(event) {
  if (event.keyCode == 13 || event.which == 13) {
    updateKeyFrame();
  }
});
$('#processButton').on('click', function() {
  callServer();
})
$(document).on('click', '.frame', function() {
  var selectedId = parseInt(this.id.split('_')[1]);
  $('#segment').val(selectedId);
  videoNode.currentTime = selectedId;
});
