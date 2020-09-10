$(document).ready(function () {
	$('button').click(function () {
		var heading = $('#heading').val();
		var content = $('#content').val();
		console.log(heading)
		console.log(content)
		$.post(
			'/api/v1/summarize_call',
			{
				"headline": heading,
				"content": content,
			},
			function (data, status) {
				alert('Data: ' + data + '\nStatus: ' + status);
			}
		);
	});
});
