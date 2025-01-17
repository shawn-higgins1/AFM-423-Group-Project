�	��^~�7@��^~�7@!��^~�7@	�d�z�?�d�z�?!�d�z�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��^~�7@��iT\4@1�
}���?A�z6�>�?I5���@Y�k$	�p?*	33333sJ@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�N@aÓ?!�g{�=B@)�5�;Nё?1��;5r@@:Preprocessing2U
Iterator::Model::ParallelMapV2y�&1��?!!*w:@)y�&1��?1!*w:@:Preprocessing2F
Iterator::Model��A�f�?!�O���C@)lxz�,C|?1!Y�B*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�q����?!%���V}-@)��0�*x?1u.�eN&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����Mb�?!'�TA�>N@)y�&1�l?1!*w@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!%t�û@)ŏ1w-!_?1%t�û@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapQ�|a2�?!Y�B��C@)Ǻ���V?1H��	,@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensora2U0*�S?!�2'�%@)a2U0*�S?1�2'�%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!H��	,�?)Ǻ���F?1H��	,�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�d�z�?I7�`q��X@QW�cyO�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��iT\4@��iT\4@!��iT\4@      ��!       "	�
}���?�
}���?!�
}���?*      ��!       2	�z6�>�?�z6�>�?!�z6�>�?:	5���@5���@!5���@B      ��!       J	�k$	�p?�k$	�p?!�k$	�p?R      ��!       Z	�k$	�p?�k$	�p?!�k$	�p?b      ��!       JGPUY�d�z�?b q7�`q��X@yW�cyO�?�"5
sequential/dense/MatMulMatMul
K�E֢?!
K�E֢?0"C
%gradient_tape/sequential/dense/MatMulMatMul�NI�2
�?!^,�8<�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�NI�2
�?!���Uu�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�NI�2
�?!���c7}�?"7
sequential/dense_1/MatMulMatMul�I y�ѝ?!�Ʃrr7�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad����?!���B��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�Wi�mĔ?!��f�l�?"!
Adam/PowPow�NI�2
�?!��p���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�NI�2
�?!�:
]��?")
sequential/CastCast�NI�2
�?!K��Qh�?Q      Y@Y�C=�C=0@a��
��T@q9�C؜W@y��R���?"�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 