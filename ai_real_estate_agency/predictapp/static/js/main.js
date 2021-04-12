$(document).ready(function(){
	console.log('b');
	$('header .hdMenu li').click(function(){
		console.log('a');
		$('.hdMenu li').removeClass('active');
		$(this).addClass('active');
	});
});