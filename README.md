# MovieRecommendation
#To start the frontend(go to the localhost link generated by the frontend)
RUN cd /frontend
RUN npm !
RUN npm run dev

#To start the backend
RUN cd /backend
RUN python config.py
RUN python initdb.py
RUN python server.py
