import os

def main():
  
    for _, filename in enumerate(os.listdir('./decentralized_exploration/results')):
        if '_0fc' in filename:
            src ='./decentralized_exploration/results/'+ filename
            dst ='./decentralized_exploration/results/'+ filename.replace('_0fc', '_100fc')
            os.rename(src, dst)
        elif '10fc' in filename:
            src ='./decentralized_exploration/results/'+ filename
            dst ='./decentralized_exploration/results/'+ filename.replace('10fc', '90fc')
            os.rename(src, dst)
        elif '20fc' in filename:
            src ='./decentralized_exploration/results/'+ filename
            dst ='./decentralized_exploration/results/'+ filename.replace('20fc', '80fc')
            os.rename(src, dst)
  
# Driver Code
if __name__ == '__main__':
      
    # Calling main() function
    main()