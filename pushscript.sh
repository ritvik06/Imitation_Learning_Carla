#!/bin/sh
echo "Please mention a choice"
echo "(1) clone a new repo"
echo "(2) push new added files"
read -p "Enter your choice : " choice  

case "$choice" in
#case 1
"1") 
	read -p "Enter the repo link : " link
	git clone link;;

#case 2
"2")
	git pull origin master
	#solve merge conflicts
	git add .
	read -p "Enter your commit message: " message
	git config user.name 'ritvik06'
	git commit -m message
	git push origin master;;
esac
