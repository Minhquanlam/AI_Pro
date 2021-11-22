<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<!-- saved from url=(0035)https://www.mail-tester.com/contact -->

<?php
	function checkspam() {
		$input= $_POST["message"];
        $command = escapeshellcmd('finaltest.py '.$input);
        $output = exec($command);
        echo $output;
	}
?>

<html xmlns="" lang="en"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
			
			<title>Check spam email</title>
			<meta name="keywords" content="mail,tester,contact,test,check,email,spam,spamassassin">
			<meta name="description" content="Feel free to contact us if you have any question or comment about our service.">
			<meta name="viewport" content="width=device-width">

			
									<link rel="icon" type="image/png" href="https://www.mail-tester.com/img/favicon.png">
			<link href="./Design/css" rel="stylesheet">
			<link rel="stylesheet" type="text/css" href="./Design/app.css">
			
			<script src="./Design/ga.js.download"></script><script type="text/javascript" src="./Design/jquery-3.4.1.min.js.download"></script>
			<script type="text/javascript" src="./Design/app.js.download"></script>

			
							<link rel="stylesheet" type="text/css" href="./Design/cookieconsent.min.css">
				<script src="./Design/cookieconsent.min.js.download"></script>
					<style></style></head>

		<body id="contact">
		
	<div id="header">
		<h1 class="title py-5 m-0">Check spam email</h1>
	</div>

<div class="bg-white">
	<div class="container py-5">
		<form class="form" action="?checkspamMG3O7oMJixl3CPZEN4D5=true" method="post" onclick = "checkspam()">


				<div class="row my-3">
					<div class="col-sm-2 text-sm-right">
						<label class="form-control-lg" for="message">Your message</label>
					</div>
					<div class="col-sm-8">
						<textarea id="message" required="required" class="form-control form-control-lg" name="message" rows="5" cols="20"></textarea>
					</div>
				</div>
				<div class="my-3 text-center form-inline justify-content-center d-none">
												<label class="form-control-lg" for="captcha">8 + 7 = </label>
						<input type="text" id="captcha" style="max-width:100px;" class="form-control form-control-lg" name="captcha">

				</div>

				<div class="text-center">
					<input type="submit" class="submit btn btn-primary btn-lg" value="Submit">
				</div>


			
		</form>
	</div>
</div>
	
		
		
		</body></html>

<?php
	if (isset($_GET['checkspamMG3O7oMJixl3CPZEN4D5'])) {
		checkspam();
	}
?>