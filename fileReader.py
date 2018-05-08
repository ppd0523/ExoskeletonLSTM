# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import glob


def read_files(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    record_defaults = [[4096], [4096], [4096]]
    raw_data = tf.decode_csv(value, record_defaults=record_defaults, field_delim=',')

    return raw_data

def input_pipeline(filenames, num_line, num_epochs=None, shuffle=False):
    # tf용 file directory load가 있다.
    fileList = glob.glob(filenames)
    fileList.sort()
    filename_queue = tf.train.string_input_producer(fileList, num_epochs=num_epochs, shuffle=shuffle, name="filename_queue")
    raw_data = read_files(filename_queue)

    x, y = tf.train.batch([raw_data[:-1], [raw_data[-1]]], batch_size=num_line)
    return x, y


### 만일 input_pipeline()이서 batch 만큼 tf.train.batch를 요청해도 file의 데이터가 부족하면
### x,y array 길이가 부족하게 생성 된다. reshape_LSTM_input에서 index 접근시 오류를 발생시킬
### 가능성이 있다.
def reshape_LSTM_input(dataX, dataY, num_data, sequence_length):

    x, y = [], []
    data_size = 0
    for i in range(num_data - sequence_length + 1):
        try:
            _x = dataX[i : i + sequence_length]
            _y = dataY[i + sequence_length - 1]
            # data_size += 1
            x.append(_x)
            y.append(_y)
        except tf.errors.OutOfRangeError:
            break

    return x, y

batch_size = 1
sequence_length = 2
total_epochs = 1


# rawX, rawY = input_pipeline("./data/data[0-9].txt", num_line=batch_size, num_epochs=total_epochs)
# trainX, trainY = reshape_LSTM_input(rawX, rawY, num_data=batch_size, sequence_length=sequence_length)

sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

list = glob.glob("./data/data[0-9].txt")
list.sort()
filename_queue = tf.train.string_input_producer(list, num_epochs=None, shuffle=False, name="filename_queue")
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
raw_data = tf.decode_csv(value, record_defaults=[[0], [0], [0]], field_delim=',')
r = tf.train.batch(raw_data, batch_size=1) # raw_data에서 한 set의 data가 생성되어야 하나...

print(sess.run(r)) # 코드 정지되는 부분

# exit(1)
#
# try:
#     while True:
#         x_batch, y_batch = sess.run([trainX, trainY])
#         print(len(y_batch), x_batch, y_batch)
#
# except tf.errors.OutOfRangeError as e:
#     print("error")
#
# finally:
#     print("end")

coord.request_stop()
coord.join(threads=threads)
sess.close()