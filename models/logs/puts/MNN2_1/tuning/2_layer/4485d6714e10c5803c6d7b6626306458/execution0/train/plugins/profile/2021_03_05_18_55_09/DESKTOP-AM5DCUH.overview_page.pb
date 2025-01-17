�	��m��7@��m��7@!��m��7@	Bןc!�?Bןc!�?!Bןc!�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��m��7@��_��4@1�y���?A_)�Ǻ�?Iis�ۄ;@Y��	L�uk?*	������F@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�� �rh�?!�F�|�B@)�q����?1����]A@:Preprocessing2U
Iterator::Model::ParallelMapV2�+e�X�?!8�0���8@)�+e�X�?18�0���8@:Preprocessing2F
Iterator::Model��y�):�?!B:؎yC@)-C��6z?1����=,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_�y?!�q�4�+@)/n��r?1�2�\�A#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�;Nё�?!��'q�N@)-C��6j?1����=@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!z} �T�@)ŏ1w-!_?1z} �T�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapHP�sג?!zo�I�!D@)Ǻ���V?1������@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!����=�?)-C��6J?1����=�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!���6��?)a2U0*�C?1���6��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Bןc!�?IO�/�X@Qgl�с��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��_��4@��_��4@!��_��4@      ��!       "	�y���?�y���?!�y���?*      ��!       2	_)�Ǻ�?_)�Ǻ�?!_)�Ǻ�?:	is�ۄ;@is�ۄ;@!is�ۄ;@B      ��!       J	��	L�uk?��	L�uk?!��	L�uk?R      ��!       Z	��	L�uk?��	L�uk?!��	L�uk?b      ��!       JGPUYBןc!�?b qO�/�X@ygl�с��?�"5
sequential/dense/MatMulMatMul�C��W�?!�C��W�?0"C
%gradient_tape/sequential/dense/MatMulMatMul4�; �1�?!gL?R�D�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul4�; �1�?!]�ݺ?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�Ahx��?!4�; �1�?"7
sequential/dense_1/MatMulMatMul�Ahx��?!g�HO���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchxJ�~�?!g�QLV��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradxJ�~�?!g9[IT�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulxJ�~�?!g�dF��?")
sequential/CastCast4�; �1�?!��k�*�?"7
sequential/dense_2/MatMulMatMul4�; �1�?!��9�%��?0Q      Y@Y�C=�C=0@a��
��T@q�+n��W@y]���?"�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�95.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 