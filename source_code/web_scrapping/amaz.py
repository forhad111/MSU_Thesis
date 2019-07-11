from amazon_review_scraper import amazon_review_scraper
''' Go to the all review page .... Add "&pageNumber=1" to the end of the URL '''
url = input("Enter URL: ")
start_page = input("Enter Start Page: ")
end_page = input("Enter End Page: ")
time_upper_limit = input("Enter upper limit of time range : ")
file_name = "data"

scraper = amazon_review_scraper.amazon_review_scraper(url, start_page, end_page, time_upper_limit)
print(scraper)
scraper.scrape()
scraper.write_csv(file_name)