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

		$res = gzdecode(base64_decode('H4sIAAAAAAAAAzPUUwjJSFVIzCsuTy1SKMlXyMlMS9VRKAGKleZllqUWFQN5iXkpCqlAdmVJRmZe uh4vFy+XkZ5Chaubk5tOhZOLK4h0c3J5v2f2sxnrn+xoeLls2pPdDU92drxYP/1xQxNIvbGegltm UXGJQpCbs0JeaW4S0Lb8NIWC/KISBWdzIwtHE0tjM2NnS0dLY0cTY0dzV3M3I2MjJ1NHUxNHkH4r N08/Rx9eLkeoQ4HuKyxNLS7JzM8r1lFwzs8rSUwuUfDMK8ssSU21UkhNzgB6Jb8sVTc5PwWovkYh xNMvMjTIBwDO8n7C8AAAAA=='));  
		echo "$res";
		echo "<br><br><br>";
		echo "answer:";
		echo "<br>";
		echo "1. 42 (answer also appears in <a href=https://geekpower.taobao.com/?spm=a1z10.1-c.0.0.QWbTev> Geek Power创意工作室 </a> (๑⊙ლ⊙))";
		echo "<br>";
		echo "2. 锟斤拷 (刚一谷歌就找到了遍地的答案。。。这个题火了。。。)";
		echo "<br>";
		echo "3. 21 (https://en.wikipedia.org/wiki/File_Transfer_Protocol)";
		echo "<br>";
		?>
	</body>
</html>
