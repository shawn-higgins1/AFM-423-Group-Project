	鵫q򶔲@鵫q򶔲@!鵫q򶔲@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-鵫q򶔲@裾庘u4@1J硑笙?AA傗菢沪?I s-Z�� @*	433333G@2U
Iterator::Model::ParallelMapV2鷡j紅搱?!	�=嵃�9@)鷡j紅搱?1	�=嵃�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!避�私=@)g甄j+鰢?1聄O#,79@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate旜_楲�?!鏋FX頸6@)9慈v緹z?1濬X頸,@:Preprocessing2F
Iterator::Model郸y�):�?!呭濬X.C@)鄿ソ羨?1      )@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice辯妿潋o?!,�4聄� @)辯妿潋o?1,�4聄� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip摡俀I�?!|a恭袾@)鷡j紅揾?1	�=嵃�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor孞�4a?!恭杮@)孞�4a?1恭杮@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!勫濬Xn9@)呛笉餠?1�4聄O#@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIL?輏篨@Q�,癏%|�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	裾庘u4@裾庘u4@!裾庘u4@      ��!       "	J硑笙?J硑笙?!J硑笙?*      ��!       2	A傗菢沪?A傗菢沪?!A傗菢沪?:	 s-Z�� @ s-Z�� @! s-Z�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qL?輏篨@y�,癏%|�?