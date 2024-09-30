from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import scipy
import cvxpy as cp
import warnings

warnings.filterwarnings("ignore")
from pyscipopt import scip
import os

app = Flask(__name__)

def get_city_quote(city, num_lines):
    first_tier_cities = ['北京市', '上海市', '广州市', '深圳市', '海南市']
    if city in first_tier_cities:
        if num_lines == 1:
            quote = 1473
        elif num_lines == 2:
            quote = 1217
        elif num_lines == 3:
            quote = 1725
    else:
        if num_lines == 1:
            quote = 1087
        elif num_lines == 2:
            quote = 898
        elif num_lines == 3:
            quote = 1273
    return quote

def convex_opt(times, k, fee1, fee2, opt, score):  # Number of workers
    n = len(times)  # Number of jobs
    std_times = times.copy()
    x = cp.Variable((n, k), boolean=True)  # Binary variable
    C = cp.Variable((n, k))

    # Objective function
    objective = cp.Minimize(cp.max(C))

    # Constraints
    constraints = [
        cp.sum(x, axis=1) == 1,  # Each job is assigned to exactly one worker
        cp.sum(x, axis=0)
        <= np.ceil(n / k) + 1,  # Each worker processes one job at a time
        C >= cp.max(
            cp.sum(cp.multiply(
                x,
                np.repeat(times.process_times.values, k).reshape(
                    (n, k))) + cp.multiply(
                        x,
                        np.repeat(times.release_times.values, k).reshape(
                            (n, k))),
                   axis=0))
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCIPY, qcp=True, verbose=False)

    # Extract and print results
    assignment_matrix = abs(np.round(x.value))
    total_working_time = np.max(C.value)
    time_consume = np.sum(abs(
        np.round(
            np.multiply(
                x.value,
                np.repeat(times.process_times.values, k).reshape(
                    (n, k))) + np.multiply(
                        x.value,
                        np.repeat(times.release_times.values, k).reshape(
                            (n, k))), 0).T),
                          axis=1)
    assign_jobs = {}
    for i in range(k):
        assign_jobs[i] = {
            'total_working_time':
            time_consume[i],
            'assigned_jobs':
            [j for item in np.argwhere(assignment_matrix.T[i]) for j in item]
        }

    dff = pd.DataFrame.from_dict(assign_jobs, orient='index')
    dff.reset_index(inplace=True)
    dff.rename(columns={'index': 'worker_id'}, inplace=True)
    dff['assigned_jobs'] = dff['assigned_jobs'].apply(
        lambda x: [std_times.jobs[i] for i in x])
    final_score = ((1 - score) * total_working_time +
                   score * total_working_time * fee2 / 60 * k)
    return dff, [
        np.ceil(total_working_time),
        np.ceil(total_working_time * fee1 / 60 * k),
        np.ceil(total_working_time * fee2 / 60 * k), final_score
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        num_people = int(request.form['num_people'])
        num_lines = num_people
        store_area = int(request.form['store_area'])
        hourly_wage = 63
        city = str(request.form['city'])
        gmv = int(request.form['gmv'])
        opt = str(request.form['opt'])
        ice_tea_machine = str(request.form['ice_tea_machine'])

        if (city in [
                '北京市', '上海市', '广州市', '深圳市', '海口市', '三亚市', '儋州市', '万宁市', '东方市'
        ]):
            price_new = 1.3 * hourly_wage
        elif city in [
                '成都市', '重庆市', '杭州市', '西安市', '武汉市', '苏州市', '郑州市', '南京市', '天津市',
                '长沙市', '东莞市', '宁波市', '佛山市', '合肥市', '青岛市'
        ]:
            price_new = 1.005 * hourly_wage
        elif city in '烟台市、无锡市、福州市、温州市、佛山市、沈阳市、济南市、厦门市、金华市、哈尔滨市、长春市、石家店市、大连市、南宁市、泉州市、嘉兴市、贵阳市、常州市、南昌市、南通市、徐州市、惠州市、太原市、绍兴市、保定市、潍坊市、临沂市、台州市、中山市、珠海市'.split('、')+\
        [x+'市' for x in ['兰州','海口','湖州','扬州','洛阳','汕头','盐城','赣州','唐山','乌鲁木齐','济宁','镇江','廊坊','咸阳','泰州','芜湖','邯郸','揭阳','南阳','呼和浩特','阜阳','江门','银川','遵义','漳州','桂林',
        '淄博','新乡','连云港','沧州','绵阳','衡阳','商丘','菏泽','信阳','襄阳','滁州','上饶','九江','宜昌','莆田','湛江','柳州','安庆','宿迁','肇庆','周口','邢台','荆州','三亚','岳阳','蚌埠','驻马店','泰安','潮州','株洲',
        '威海','六安','常德','安阳','宿州','黄冈','德州','宁德','聊城','宜春','渭南','清远','南充']]:
            price_new = 1 * hourly_wage
        else:
            price_new = hourly_wage * 0.97

        # 创建一个数据框作为 convex_opt 函数的参数
        jobs = [
            '打烊准备', '茶饮机', '奶昔机', '蒸汽机', '封口机', '器具清洗', '吧台', '地面', '后厨',
            '打烊收尾'
        ]
        process_times_base = [0.5, 23., 3, 7, 5, 24, 10, 0.26, 23, 10]
        release_times = [0] * 10
        系数 = 1.0 * (0.6)**(max(0, min(num_lines - 1, 1)))
        store_config = [
            1, num_lines, 系数 * num_lines, 系数 * num_lines, 系数 * num_lines,
            系数 * num_lines, 系数 * num_lines, store_area, 系数 * num_lines, 1
        ]
        process_times = np.multiply(store_config, process_times_base)
        process_times[-3] = np.ceil(process_times[-3] / 10 + 0.5) * 10
        standard_times = pd.DataFrame({
            'jobs': jobs,
            'base_time': process_times_base,
            'store_config': store_config,
            'process_times': process_times,
            'release_times': release_times
        })
        standard_times[
            'total_times'] = standard_times.process_times + standard_times.release_times

        assignment, values, st_time = run(num_people, num_lines, store_area,
                                           price_new, opt)

        if (num_lines > 1):
            if (gmv < 700000) and (store_area < 100):
                values[2] = values[2] * 0.825
            elif (gmv < 700000) and (store_area < 140):
                values[2] = values[2] * 0.85
            elif (gmv < 700000) and (store_area < 210):
                values[2] = values[2] * 0.875
            elif (gmv < 700000) and (store_area < 350):
                values[2] = values[2] * 0.9
            elif (gmv < 700000) and (store_area < 1000):
                values[2] = values[2] * 0.9
        else:
            if gmv < 300000:
                values[2] = values[2] * 0.964
            elif gmv < 500000:
                values[2] = values[2] * 0.964
            elif gmv < 700000:
                values[2] = values[2] * 0.964

        # 计算月报价并加上城市和冰茶机费用
        base_monthly_cost = values[2] * 30.5
        if opt == '是':
            city_quote = get_city_quote(city, num_lines)
            base_monthly_cost += city_quote
        if ice_tea_machine == '是':
            base_monthly_cost += 450

        return render_template('result.html', assignment=assignment, values=values, st_time=st_time, monthly_cost=base_monthly_cost)

    elif request.method == 'GET':
        return render_template('index.html')

def run(人数, 动线数, 面积, 时薪, 选项):
    jobs = [
        '打烊准备', '茶饮机', '奶昔机', '蒸汽机', '封口机', '器具清洗', '吧台', '面积', '后厨', '打烊收尾'
    ]
    process_times_base = [0.5, 23., 3, 7, 5, 24, 10, 0.26, 23, 10]
    release_times = [0] * 10
    系数 = 1.0 * (0.6)**(max(0, min(动线数 - 1, 1)))
    store_config = [
        1, 动线数, 系数 * 动线数, 系数 * 动线数, 系数 * 动线数, 系数 * 动线数, 系数 * 动线数, 面积, 系数 * 动线数,
        1
    ]
    process_times = np.multiply(store_config, process_times_base)
    process_times[-3] = np.ceil(process_times[-3] / 10 + 0.5) * 10
    standard_times = pd.DataFrame({
        'jobs': jobs,
        'base_time': process_times_base,
        'store_config': store_config,
        'process_times': process_times,
        'release_times': release_times
    })
    standard_times[
        'total_times'] = standard_times.process_times + standard_times.release_times
    first = standard_times.head(1)
    last = standard_times.tail(1)
    rest_sorted = standard_times.iloc[1:-1].sort_values('total_times',
                                                        ascending=False)
    standard_times = pd.concat([first, rest_sorted, last])
    standard_times =standard_times.reset_index(drop=True)

    assigment, values = convex_opt(standard_times, 人数, 50, 时薪, 选项, 0.9)
    st_time = standard_times.rename(
        {
            'jobs': '任务事项',
            'base_time': '单位清洗时间',
            'store_config': '门店配置',
            'process_times': '总清洗时间',
            'release_times': '等待时间',
            'total_times': '总时间'
        },
        axis=1)
    print(f'单店打烊预估时长：{values[0]}')
    print(f'单店月报价({时薪}/h): {values[2]*30.5} ')
    return assigment, values, st_time

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4200)
