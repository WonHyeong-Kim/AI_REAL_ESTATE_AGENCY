$(document).ready(function(){
	$('header .hdMenu li').click(function(){
		$('.hdMenu li').removeClass('active');
		$(this).addClass('active');
	});
});