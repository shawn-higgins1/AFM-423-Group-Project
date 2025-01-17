�	�f�"�7@�f�"�7@!�f�"�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�f�"�7@R�8��}4@1�B�Y���?A��@��Ǩ?I�-���@*	�����?J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��ׁsF�?!�m۶m�B@)��d�`T�?1�0�0A@:Preprocessing2U
Iterator::Model::ParallelMapV2S�!�uq�?!b�a�9@)S�!�uq�?1b�a�9@:Preprocessing2F
Iterator::Model8��d�`�?!=��<��B@)9��v��z?11�0�(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�J�4�?!      0@)9��v��z?11�0�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���{�?!�0�0O@)a��+ei?1�y��y�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!=��<��@)ŏ1w-!_?1=��<��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap^K�=��?!�a�aD@)��_�LU?1�<��<�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!��y��y�?)����MbP?1��y��y�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��H�}M?!ܶm۶m�?)��H�}M?1ܶm۶m�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 85.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�13.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIח��X@QB�>
��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	R�8��}4@R�8��}4@!R�8��}4@      ��!       "	�B�Y���?�B�Y���?!�B�Y���?*      ��!       2	��@��Ǩ?��@��Ǩ?!��@��Ǩ?:	�-���@�-���@!�-���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qח��X@yB�>
��?�"5
sequential/dense/MatMulMatMul_5�v~ �?!_5�v~ �?0"C
%gradient_tape/sequential/dense/MatMulMatMul�]S���?!�ɝ��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�]S���?!�x��
w�?0"7
sequential/dense_1/MatMulMatMul�]S���?!ÓxK8��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulQ�@䓝?!M�zӴ��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul_5�v~ �?!��W��K�?"7
sequential/dense_2/MatMulMatMul_5�v~ �?!��4q���?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�]S���?!bI��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�]S���?!��U�)�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�]S���?!n
d0��?Q      Y@Y�C=�C=0@a��
��T@qH���W@y	�y�v��?"�
both�Your program is POTENTIALLY input-bound because 85.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�13.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 