
import pytest

import tempfile

from piscat.Tutorials.startup import *
import piscat.Tutorials.startup


def test_parse_args_working_dir(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        mocker.patch.object(piscat.Tutorials.startup.sys, 'argv', ['the_script', '.'] )
        target = parse_args()
        assert Path(target).absolute() == Path.cwd()


def test_parse_args_no_working_dir(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        mocker.patch.object(piscat.Tutorials.startup.sys, 'argv', ['the_script'])
        mocker.patch('piscat.Tutorials.startup.confirm', return_value=True)
        target = parse_args()
        assert Path(target).absolute() == Path.cwd()


def test_parse_args_no_working_dir_not_confirmed(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        mocker.patch.object(piscat.Tutorials.startup.sys, 'argv', ['the_script'])
        mocker.patch('piscat.Tutorials.startup.confirm', return_value=False)
        with pytest.raises(Exception):
            parse_args()


def test_parse_args_dir_somewhere_else(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        mocker.patch.object(piscat.Tutorials.startup.sys, 'argv', ['the_script', 'abc'])
        target = parse_args()
        assert Path(target).absolute() == Path('abc').absolute()


def test_parse_args_too_many_args_raises_exception(mocker):
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        mocker.patch.object(piscat.Tutorials.startup.sys, 'argv', ['the_script', 'abc', 'cba'])
        with pytest.raises(Exception):
            parse_args()


def test_find_jupyter_finds_it(mocker):
    mocker.patch('piscat.Tutorials.startup.shutil.which', return_value='jup_location')
    jupyter = find_jupyter()
    assert jupyter == 'jup_location notebook'


def test_find_jupyter_does_not_find_it(mocker):
    mocker.patch('piscat.Tutorials.startup.shutil.which', return_value=None)
    with pytest.raises(Exception):
        find_jupyter()


def test_functional_default(mocker):

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        target = Path(tmpdir) / 'Tutorials' / 'JupyterFiles' / 'Tutorial1.ipynb'
        mocker.patch('piscat.Tutorials.startup.start_server')
        mocker.patch('piscat.Tutorials.startup.find_jupyter', return_value='/dummy/jupyter')
        mocker.patch.object(piscat.Tutorials.startup.sys, 'argv', ['the_script', '.'])
        assert not exists(target)
        start_tutorials()
        assert exists(target)


def test_functional_do_not_copy_when_dir_exists(mocker):

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        os.mkdir('abc')
        target = Path(tmpdir) / 'Tutorials' / 'JupyterFiles' / 'Tutorial1.ipynb'
        mocker.patch('piscat.Tutorials.startup.start_server')
        mocker.patch('piscat.Tutorials.startup.find_jupyter', return_value='/dummy/jupyter')
        mocker.patch.object(piscat.Tutorials.startup.sys, 'argv', ['the_script', 'abc'])
        assert not exists(target)
        start_tutorials()
        assert not exists(target)


def test_functional_if_not_jupyter_found_no_server_called(mocker):

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        target = Path(tmpdir) / 'Tutorials' / 'JupyterFiles' / 'Tutorial1.ipynb'
        mock_server = mocker.patch('piscat.Tutorials.startup.start_server')
        mocker.patch('piscat.Tutorials.startup.shutil.which', return_value=None)
        mocker.patch.object(piscat.Tutorials.startup.sys, 'argv', ['the_script', '.'])
        with pytest.raises(Exception):
            start_tutorials()
        assert exists(target)
        assert not mock_server.called