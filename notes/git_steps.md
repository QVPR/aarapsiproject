Commands in terminal/bash for linux:
```
git config --global user.name "Sam Smith"
```
```
git config --global user.email "samsmith@qut.edu.au"
```
```
git config --global core.editor nano
```
```
cd ~/.ssh
```
```
ssh-keygen -t ed25519 -C $(git config --get user.email)
```
Provide a recogniseable name (we will use it later)
Feel free to provide a passphrase but every time you use the key (i.e. github push/pull) you will need to enter it.
For the tutorial, I have used the name 'github_key' and no passphrase
```
eval "$(ssh-agent -s)"
```
```
ssh-add ~/.ssh/github_key
```
```
cat ~/.ssh/github_key.pub
```
Now copy-paste the contents, should look like:
```
ssh-ed25519 sdahdasjhdkjashdkjashdkhasdjjkbasbfbsdfbsdffe samsmith@qut.edu.au
```
Now go to your github account, then settings, then SSH and GPG keys, then Add new SSH key.
Paste in the contents, and name it something recognisable such as 'hpc_key', save it.
Go to wherever you plan on downloading the repository:
```
mkdir ~/repositories; cd ~/repositories
```

