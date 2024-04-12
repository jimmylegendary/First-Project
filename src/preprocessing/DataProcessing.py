class DataProcessing(object):
    def select_mortality_ahi_subject(self, data):
        
        ## including eds 
        quality_index = [id for id, value in data.iterrows() if value['ahi']>=5]
        df = data.iloc[quality_index,:]
        df.reset_index(drop=True, inplace=True)

        ## grouping subjective sleep
        alive_index = [id for id, value in df.iterrows() if (value['vital']==1 or (value['vital']==0 and value['censdate']>365*15))]
        df['vital'][alive_index]=1
        alive = df.iloc[alive_index,:]        
        alive.reset_index(drop=True, inplace=True)
        alive['vital']=0

        deceased_index = [id for id, value in df.iterrows() if (value['vital']==0 and value['censdate']<=365*15)]
        deceased = df.iloc[deceased_index,:]
        deceased.reset_index(drop=True, inplace=True)
        deceased['vital']=1
        
        print('alive:', len(alive), '\n deceased:', len(deceased))

        return alive, deceased