<html lang="en">
    <head>
        <meta http-equiv="Content-Type"content="text/html; charset=UTF-8">
    </head>
	<body>
		<?php

		if (function_exists("gzdecode")) {
			echo "gzdecode OK, <br>";
		} else {
			echo "gzdecode no OK, <br>";
			function gzdecode($data) { 
				return gzinflate(substr($data,10,-8)); 
			} 
		}
		$url = "https://gist.githubusercontent.com/zealic/38510fd8ecd1be75924a/raw/27ac7ef5d3d7dd70d83e3cd2b5049e239e4dba72/0x8B1F";
		#$url = "https://gist.githubusercontent.com/zealic/38510fd8ecd1be75924a/raw/27ac7ef5d3d7dd70d83e3cd2b5049e239e4dba72/0x8B1F";
		$que = file_get_contents($url);

		echo $que;
		
		echo "<br><br><br> answer1 is:<br>";
		$res = base64_decode($que);
		echo base_convert(ord($res[0]),10,16);
		echo base_convert(ord($res[1]),10,16);
		echo "<br>";
		echo "$res";
		echo "<br><br>";
		$res = gzdecode($res);
		echo "$res";
		
		echo "<br><br><br> answer2 is:<br>";
		$que = 	"/Td6WFoAAATm1rRGAgAhARYAAAB0L+Wj4AFRAQVdABGICqfJeJWaXR0sjwRpdS7E +6IpZ7jEgwnjNXF1I+Am53CER4uNimC3eWr2cLkUF4wS790hhcFJDPibD3E6vykU wfVcZtkxf49xh2iethSwFIx3fmnT/Qzol3c2NzHNLm08op6oj6c6IymXQoTGUfjt QMm4Q8yuiGCX2iLAqUsg/SFnJNb1G4QuOWCVZSQwrNnHoZnAsJjXKdxW82xVSI5H T2GH19HrQWj7mAIv2hEG7rHr7Lvc5KrtVPN4+jGZQPbvWM4Mh+Sjx/wTyxQ14MW/ FgSVKDir3PXIfM3/4bPggO6EQc3KOgNzQl+E5ePAoI0wNh+KEohXqYVt4giRNm52 ypAgAAAAAAC8TejHH4hG6QABoQLSAgAAiuAM5LHEZ/sCAAAAAARZWg==";
		$res = base64_decode($que);
		echo base_convert(ord($res[0]),10,16);
		echo base_convert(ord($res[1]),10,16);
		echo "<br>";
		echo "$res";		
		echo "<br><br>";		
		$fileName = "code-lover.tar.xz";
		if ($fp = fopen($fileName, "w+")) {
			if (@fwrite($fp, $res)) {
				fclose($fp);
				return true;
			} else {
				echo "NANI!!!";
				fclose($fp);
				return false;
			} 
		}  		
		echo "$res";

		?>
	</body>
</html>
