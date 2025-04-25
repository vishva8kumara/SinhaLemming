
const fs = require('fs');
//const utils = require("./utils");
const x = '­';

async function analyze() {
	const lemming = (await fs.readFileSync('../words.keys.json')).toString().split('\n');
	//
	let output = {};
	for (let line of lemming) {
		if (line.indexOf(' ') > -1) {
			line = line.replace(/[\u00AD\u200B-\u200D\uFEFF]/g, '').split(' ');
			const stem = line.shift();
			output[ stem ] = '<'+stem+'>';
			for (let wrd of line)
				if (wrd != stem)
				output[ wrd ] = '<'+stem+'>';
		}
	}
	//
	fs.writeFile('./input.json', JSON.stringify(output, null, 4), function(err) {
		if (err) {
			return console.log(err);
		}
		console.log('input.json is saved');
	});

}
analyze();
