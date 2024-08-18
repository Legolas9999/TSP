from TSPfunc import dist
from math import ceil

def iterate_for_max_lamuda(cities_list:list) -> tuple:
    #每个城市所对应的最大lamuda
    max_lamuda_for_every_absebt_city = []

    #记录每个消失城市的最大lamuda所对应的左边，右边城市索引
    left_and_right_index_for_every_absent_city = []

    num_of_city = len(cities_list)

    #左边和右边等待遍历城市个数
    num_of_city_iter = num_of_city - 2


    #遍历消失城市
    for index_of_absent_city in range(num_of_city):
        #该城市的最大lamuda
        max_lamuda_for_current_absent_city = 0.0

        #初始化左右边城市列表
        left_city_list = []
        right_city_list = []

        #计算左边，如果是最后一个
        if index_of_absent_city == num_of_city - 1:
            for i in range(num_of_city_iter):
                left_city_list.append(i)  
        
        #计算左边，其他位置
        else:
            for i in range(num_of_city_iter+1):
                #遍历到自己则跳过
                if i != index_of_absent_city:
                    left_city_list.append(i)
        
        #计算右边，如果是前两个,右边城市列表相同
        if index_of_absent_city == 1 or index_of_absent_city == 0:
            for i in range(2,num_of_city):
                right_city_list.append(i)

        #计算右边，其他位置的
        else:
            for i in range(1,num_of_city):
                #遍历到自己跳过
                if i != index_of_absent_city:
                    right_city_list.append(i)
        #print(left_city_list,right_city_list)


        
        
        #########开始计算lamuda########


        #从遍历左边列表开始
        for i in range(num_of_city_iter):
            for j in range(i,num_of_city_iter):

                #当前lamuda
                current_lamuda = 0.5*(dist(left_city_list[i],index_of_absent_city,cities_list) + \
                    dist(index_of_absent_city,right_city_list[j],cities_list))
                
                #当前消失城市的最大lamuda
                if current_lamuda > max_lamuda_for_current_absent_city:
                    max_lamuda_for_current_absent_city = current_lamuda
                    #并记录当前左右城市索引
                    temp_max = [left_city_list[i],right_city_list[j]]
        
        #存储每个城市的最大lamuda            
        max_lamuda_for_every_absebt_city.append(max_lamuda_for_current_absent_city) 
        #储存每个城市的最大lamuda 时对应的左右城市索引
        left_and_right_index_for_every_absent_city.append(temp_max)
                    
    #print(max_lamuda_for_every_absebt_city)          
    ##加一再向上取整       
    return ceil( max(max_lamuda_for_every_absebt_city) + 1), max_lamuda_for_every_absebt_city ,left_and_right_index_for_every_absent_city



        
         

        


