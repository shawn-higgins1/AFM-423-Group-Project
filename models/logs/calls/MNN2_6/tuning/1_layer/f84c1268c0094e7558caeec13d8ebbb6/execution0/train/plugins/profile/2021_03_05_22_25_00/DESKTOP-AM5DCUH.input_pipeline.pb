	gsꠔ@gsꠔ@!gsꠔ@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-gsꠔ@l=C8f�0@1豸吓?AbX9慈�?I夝/偲恬?*	翁烫虒G@2U
Iterator::Model::ParallelMapV2蘛K�=�?!痳涱�<<@)蘛K�=�?1痳涱�<<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat漓�<,詩?!vJa{+�;@)篒+噯?1%聊Z7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap呛笉饐?!^谈4y�7@)捤H縸}?1鍻[h罀.@:Preprocessing2F
Iterator::Modeln���?!也!娢D@) 襬_蝭?1殄O[h�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice褚Mbp?!誋2� @)褚Mbp?1誋2� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)\徛�(�?!.M揆u1M@)-C脞6j?1"娢-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor孞�4a?!F檴鐨�@)孞�4a?1F檴鐨�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIr璈u肵@Q鼺舚�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	l=C8f�0@l=C8f�0@!l=C8f�0@      ��!       "	豸吓?豸吓?!豸吓?*      ��!       2	bX9慈�?bX9慈�?!bX9慈�?:	夝/偲恬?夝/偲恬?!夝/偲恬?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qr璈u肵@y鼺舚�?