DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -z "$PYTHONPATH" ]
then
	export PYTHONPATH=$DIR
else
	export PYTHONPATH=$PYTHONPATH:$DIR
fi
