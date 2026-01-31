#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 2023

@author: lefteris

@subject: Dashboard class for the queries
"""

import pandas as pd


class DashboardQueries:
    def __init__(self, conn):
        self.conn = conn
    
    #filters
    def filters_exp_cd(self):
        query_ft_expence_code = "SELECT DISTINCT(expense_code) FROM expenses;"
        result_ft_expence_code = pd.read_sql_query(query_ft_expence_code, self.conn)
        result_ft_expence_code = list(result_ft_expence_code.expense_code)
        return result_ft_expence_code

    def filters_dept(self):
        query_result_ft_department = "SELECT DISTINCT(department) FROM expenses;"
        result_result_ft_department = pd.read_sql_query(query_result_ft_department, self.conn)
        result_result_ft_department = list(result_result_ft_department.department)
        return result_result_ft_department
    
    def filters_cpn(self):
        query_ft_company = "SELECT DISTINCT(company) FROM expenses;"
        result_ft_company = pd.read_sql_query(query_ft_company, self.conn)
        return result_ft_company
    
    def filters_dt(self):
        query_ft_start_date = "SELECT MIN(date) FROM expenses;"
        query_ft_end_date = "SELECT MAX(date) FROM expenses;"
        result_ft_start_date = pd.read_sql_query(query_ft_start_date , self.conn)

        query_ft_end_date = "SELECT MAX(date) FROM expenses;"
        result_ft_end_date = pd.read_sql_query(query_ft_end_date , self.conn)
        
        return result_ft_start_date, result_ft_end_date
    
    def filters_ditr(self):
        query_ft_district = "SELECT DISTINCT(district) FROM expenses;"
        result_ft_district = pd.read_sql_query(query_ft_district, self.conn)
        result_ft_district = list(result_ft_district.district)

        return result_ft_district
    
    def common_filtering(self, start_date, end_date, selected_expense_code, selected_department,
                         company_filter, selected_district):
        return f"""
            WHERE date BETWEEN '{str(start_date)}' AND '{str(end_date)}'
            AND expense_code IN ({', '.join(map(lambda x: f"'{x}'", selected_expense_code))})
            AND department IN ({', '.join(map(lambda x: f"'{x}'", selected_department))})
            AND district IN ({', '.join(map(lambda x: f"'{x}'", selected_district))})
            AND company LIKE '%{company_filter}%'
        """

    def table(self, start_date, end_date, selected_expense_code, selected_department, company_filter,selected_district, limit):
        filters = self.common_filtering(start_date, end_date, selected_expense_code,
                                        selected_department,company_filter, selected_district)
        df_query = f"""
            SELECT * FROM expenses
            {filters}
            LIMIT {limit}
        """
        df = pd.read_sql_query(df_query, self.conn)
        return df
    
    def district(self, start_date, end_date, selected_expense_code, selected_department, company_filter, selected_district):
        filters = self.common_filtering(start_date, end_date, selected_expense_code,
                                        selected_department, company_filter, selected_district)
        df_query = f"""
            SELECT district, sum(amount) AS amount FROM expenses
            {filters}
            GROUP BY district
        """
        df = pd.read_sql_query(df_query, self.conn)
        return df
    
    def query_total_amount(self, start_date, end_date, selected_expense_code,
                           selected_department, company_filter, selected_district):
        filters = self.common_filtering(start_date, end_date, selected_expense_code,
                                        selected_department, company_filter, selected_district)
        total_amount_query = f"""
            SELECT SUM(amount) AS total_amount FROM expenses
            {filters}
        """
        total_amount_result = pd.read_sql_query(total_amount_query, self.conn)
        return total_amount_result['total_amount'].iloc[0]
    
    def query_total_number(self, start_date, end_date, selected_expense_code,
                           selected_department, company_filter, selected_district):
        filters = self.common_filtering(start_date, end_date,
                                        selected_expense_code, selected_department, company_filter, selected_district)
        
        id_number_query = f"""
        SELECT COUNT(id) AS id_number FROM expenses
        {filters}
        """
        id_number_result = pd.read_sql_query(id_number_query, self.conn)
        return id_number_result['id_number'].iloc[0]
    
    def query_amount_by_month(self,start_date, end_date, selected_expense_code,
                              selected_department, company_filter, selected_district):
        filters = self.common_filtering(start_date, end_date, selected_expense_code,
                                        selected_department, company_filter, selected_district)
        
        line_plot_1_query = f"""
        SELECT strftime('%Y-%m', date) AS month, SUM(amount) AS total_amount FROM expenses
        {filters}
        GROUP BY month
        """
        line_plot_1_result = pd.read_sql_query(line_plot_1_query, self.conn)
        return line_plot_1_result
    
    def query_number_by_month(self,start_date, end_date, selected_expense_code,
                              selected_department, company_filter, selected_district):
        filters = self.common_filtering(start_date, end_date, selected_expense_code,
                                        selected_department, company_filter, selected_district)
        
        line_plot_2_query = f"""
        SELECT strftime('%Y-%m', date) AS month, COUNT(id) AS total_number FROM expenses
        {filters}
        GROUP BY month;"""
        line_plot_2_result = pd.read_sql_query(line_plot_2_query, self.conn)
        return line_plot_2_result
    
    def query_amount_by_month_department(self,start_date, end_date, selected_expense_code,
                              selected_department, company_filter, selected_district):
        filters = self.common_filtering(start_date, end_date, selected_expense_code,
                                        selected_department, company_filter, selected_district)
        
        line_plot_2_query = f"""
        SELECT strftime('%Y-%m', date) AS month, department, SUM(amount) AS total_amount FROM expenses
        {filters}
        GROUP BY department, month;"""
        line_plot_2_result = pd.read_sql_query(line_plot_2_query, self.conn)
        return line_plot_2_result
    
    
    def query_exp_cd_prc(self,start_date, end_date, selected_expense_code,
                         selected_department, company_filter, selected_district):
        filters = self.common_filtering(start_date, end_date, selected_expense_code,
                                        selected_department, company_filter, selected_district)
        query_1 = f"""SELECT expense_code, SUM(amount) AS total_amount FROM expenses
        {filters}
        GROUP BY expense_code;"""
        result_1 = pd.read_sql_query(query_1, self.conn)
        return result_1
    
    def query_prg_prc(self,start_date, end_date, selected_expense_code,
                      selected_department, company_filter, selected_district):
        filters = self.common_filtering(start_date, end_date, selected_expense_code,
                                        selected_department, company_filter, selected_district)
        query_2 = f"""SELECT department, SUM(amount) AS total_amount FROM expenses
        {filters}
        GROUP BY department;"""
        result_2 = pd.read_sql_query(query_2, self.conn)
        return result_2
    
    
    
    
    
    
    
    
    
    
    
    
    


