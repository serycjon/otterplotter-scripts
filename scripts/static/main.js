var colors = ['red', 'orange', 'white', 'green', 'lime'];
var neutral_id = 2;

function cycle_up(color) {
    var i = colors.indexOf(color);
    if (i < 0) {
	i = neutral_id;
    }
    new_i = Math.min(colors.length - 1, i + 1);
    return colors[new_i]

}

function cycle_down(color) {
    var i = colors.indexOf(color);
    if (i < 0) {
	i = neutral_id;
    }
    new_i = Math.max(0, i - 1);
    return colors[new_i]
}

function click_fn(i) {
    return function(event) {
	var elem = document.getElementById(i);
	var color = elem.style.borderColor;
	if (event.ctrlKey) {
	    elem.style.borderColor=cycle_down(color);
	} else {
	    elem.style.borderColor=cycle_up(color);
	}
    }
	
// .style.backgroundColor='0';
}

function get_score(elem) {
    var i = colors.indexOf(elem.style.borderColor);
    if (i < 0) {
	i = neutral_id;
    }
    return i - neutral_id;
}

function populate_form() {
    form = document.getElementById('form');

    drawings = document.getElementsByClassName("drawing");
    Array.prototype.forEach.call(drawings, function (drawing) {
	input = document.createElement('input');
	input.setAttribute('name', 'd' + drawing.id);
	input.setAttribute('value', get_score(drawing));
	input.setAttribute('type', 'hidden');

	form.appendChild(input);
    });
    return true;
}
