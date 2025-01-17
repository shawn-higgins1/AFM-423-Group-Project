�	V���]P@V���]P@!V���]P@	 5���? 5���?! 5���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6V���]P@V]�LO@1�~NA~�?A9��m4��?I��oH#@Y�,��o�?*	gffff&L@2U
Iterator::Model::ParallelMapV2����Mb�?!o�ޒOk<@)����Mb�?1o�ޒOk<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX9��v��?!��G��;@)Zd;�O��?1`$�1!m4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!9l�&�D7@)Ǻ����?1������3@:Preprocessing2F
Iterator::Model�z6�>�?!��r�(D@)S�!�uq{?1�����'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!o�ޒOk@)����Mbp?1o�ޒOk@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�J�4�?![�i��M@)ŏ1w-!o?1��S���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!��S���
@)ŏ1w-!_?1��S���
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�o_��?!ߍ��=@)a2U0*�S?1���/@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 95.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9 5���?I��|'o�X@Q)R���;�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	V]�LO@V]�LO@!V]�LO@      ��!       "	�~NA~�?�~NA~�?!�~NA~�?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	��oH#@��oH#@!��oH#@B      ��!       J	�,��o�?�,��o�?!�,��o�?R      ��!       Z	�,��o�?�,��o�?!�,��o�?b      ��!       JGPUY 5���?b q��|'o�X@y)R���;�?�"5
sequential/dense/MatMulMatMul��u,̦?!��u,̦?0"C
%gradient_tape/sequential/dense/MatMulMatMulh�_��C�?!~�5�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�,�T�?!�}c�ݻ?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�,�T�?!/��W!�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamh�_��C�?!\�t���?"!
Adam/PowPowh�_��C�?!�� �*�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradh�_��C�?!���~���?"7
sequential/dense_1/MatMulMatMulh�_��C�?!�x6�:�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�h�?!���}!�?0")
sequential/CastCast�h�?!�d�X��?Q      Y@Y��/Ċ�0@a�	�N]�T@q�[[�T@y2l��˩�?"�
both�Your program is POTENTIALLY input-bound because 95.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�82.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 